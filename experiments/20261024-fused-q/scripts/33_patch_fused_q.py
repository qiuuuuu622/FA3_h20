"""FA3_INT8_QK_FUSED_Q (on top of 29_patch_int8qk_bf16pv.py): Q quantization inside the kernel.

ptr_Q points to BF16 Q. Each consumer warp owns 16 Q rows of its GMMA fragment (warp w -> tile rows 16w..16w+15):
uint4 gmem load (8 bf16/lane) -> FWHT over 256 (3 in-thread + 5 shfl_xor stages) * 1/16 -> per-row absmax int8
-> smem_q (swizzled); the row scale goes straight to the qs registers via the S-fragment row coordinates.
Producer skips the Q TMA; consumers skip barrier_Q and sync on the QueryRotated named barrier instead.
"""
import pathlib,sys
H=pathlib.Path(sys.argv[1])
def sub(s,old,new,count=1):
    n=s.count(old);assert n==count,(old[:90],n);return s.replace(old,new)
m=(H/'mainloop_fwd_sm90_tma_gmma_ws.hpp').read_text()
assert 'Int8_QK' in m and 'FA3_INT8_QK_FUSED_Q' not in m
m=sub(m,"    static constexpr bool Int8_QK = FA3_INT8_QK && !Is_FP8;\n",
"""    static constexpr bool Int8_QK = FA3_INT8_QK && !Is_FP8;
#ifndef FA3_INT8_QK_FUSED_Q
#define FA3_INT8_QK_FUSED_Q 0
#endif
    // Q is BF16 in gmem and quantized (Hadamard + per-row int8) by the consumer warps.
    static constexpr bool Int8_QK_FusedQ = Int8_QK && FA3_INT8_QK_FUSED_Q;
""")
m=sub(m,"        if (Use_TMA_Q && thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }",
        "        if (Use_TMA_Q && !Int8_QK_FusedQ && thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }")
m=sub(m,"""            if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);""",
"""            if (!Int8_QK_FusedQ && (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);""")
m=sub(m,"""        if constexpr (!AppendKV) {
            barrier_Q.wait(work_idx % 2);
        } else {""","""        if constexpr (!AppendKV) {
            if constexpr (!Int8_QK_FusedQ) { barrier_Q.wait(work_idx % 2); }
        } else {""")
old="""        if constexpr (Int8_QK) {
            float const* qs_ptr = params.ptr_q_descale + bidh * get<1>(params.stride_q_descale) + m_block * kBlockM;
            #pragma unroll
            for (int mi = 0; mi < kI8Rows; ++mi) { qs_i8[mi] = __ldg(qs_ptr + get<0>(tScS_i8_rc(mi, _0{}))); }
        }
"""
new="""        if constexpr (Int8_QK && !Int8_QK_FusedQ) {
            float const* qs_ptr = params.ptr_q_descale + bidh * get<1>(params.stride_q_descale) + m_block * kBlockM;
            #pragma unroll
            for (int mi = 0; mi < kI8Rows; ++mi) { qs_i8[mi] = __ldg(qs_ptr + get<0>(tScS_i8_rc(mi, _0{}))); }
        } else if constexpr (Int8_QK_FusedQ) {
            static_assert(kBlockM == NumMmaThreadsQK / cutlass::NumThreadsPerWarp * 16 && kHeadDim == 256);
            Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
            int const lane = thread_idx % cutlass::NumThreadsPerWarp;
            int const row0 = (thread_idx / cutlass::NumThreadsPerWarp) * 16;
            int64_t const q_row_stride = get<0>(params.stride_Q);
            cutlass::bfloat16_t const* q_base = reinterpret_cast<cutlass::bfloat16_t const*>(params.ptr_Q)
                + (int64_t(seqlen_info.offset_q) + int64_t(m_block) * kBlockM) * q_row_stride
                + int64_t(bidh) * get<2>(params.stride_Q) + int64_t(Varlen && params.cu_seqlens_q ? 0 : bidb) * get<3>(params.stride_Q);
            #pragma unroll
            for (int mi = 0; mi < kI8Rows; ++mi) { qs_i8[mi] = 0.f; }
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int const row = row0 + j;
                float x[8];
                uint4 raw = make_uint4(0, 0, 0, 0);
                if (m_block * kBlockM + row < seqlen_q) {
                    raw = __ldg(reinterpret_cast<uint4 const*>(q_base + row * q_row_stride) + lane);
                }
                cutlass::bfloat16_t const* xb = reinterpret_cast<cutlass::bfloat16_t const*>(&raw);
                #pragma unroll
                for (int e = 0; e < 8; ++e) { x[e] = float(xb[e]); }
                // Sylvester FWHT: element index = lane * 8 + e; bits 0-2 in-thread, bits 3-7 across lanes.
                #pragma unroll
                for (int h = 1; h < 8; h <<= 1) {
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        if (!(e & h)) { float const a = x[e], b = x[e + h]; x[e] = a + b; x[e + h] = a - b; }
                    }
                }
                #pragma unroll
                for (int msk = 1; msk < 32; msk <<= 1) {
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float const p = __shfl_xor_sync(0xffffffff, x[e], msk);
                        x[e] = (lane & msk) ? p - x[e] : x[e] + p;
                    }
                }
                float amax = 0.f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) { x[e] *= 0.0625f; amax = fmaxf(amax, fabsf(x[e])); }
                #pragma unroll
                for (int msk = 1; msk < 32; msk <<= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, msk)); }
                float const s = __fdiv_rn(fmaxf(amax, 1e-10f), 127.f);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sQ_pi(row, lane * 8 + e) = int8_t(max(-127, min(127, __float2int_rn(__fdiv_rn(x[e], s)))));
                }
                #pragma unroll
                for (int mi = 0; mi < kI8Rows; ++mi) { if (get<0>(tScS_i8_rc(mi, _0{})) == row) { qs_i8[mi] = s; } }
            }
            // Generic-proxy smem writes -> visible to GMMA (async proxy) for the whole QK warpgroup set.
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK, static_cast<uint32_t>(FwdNamedBarriers::QueryRotated) /*id*/);
        }
"""
m=sub(m,old,new)
(H/'mainloop_fwd_sm90_tma_gmma_ws.hpp').write_text(m)
print('patched')
