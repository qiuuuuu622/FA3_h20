"""FA3_INT8_QK: Element=bf16 -> Q/K int8 (s8s8->s32 WGMMA), PV stays BF16.

S_f32 = float(S_s32) * q_scale[h, m] * k_scale[h_k, n] (ptr_q_descale/ptr_k_descale reused as per-token arrays,
head stride = stride_*_descale[1]). Only the IntraWGOverlap branch is patched (BF16 D256 uses it).
Without -DFA3_INT8_QK=1 the QK accumulator is bound by reference, so FP8/BF16 builds see the same tensor.
"""
import pathlib,sys
H=pathlib.Path(sys.argv[1])
def sub(s,old,new,count=1):
    n=s.count(old);assert n==count,(old[:80],n);return s.replace(old,new)
m=(H/'mainloop_fwd_sm90_tma_gmma_ws.hpp').read_text()
assert 'FA3_INT8_QK' not in m
m=sub(m,"    using ElementQK = std::conditional_t<Mixed_QK, cutlass::bfloat16_t, Element>;\n",
"""#ifndef FA3_INT8_QK
#define FA3_INT8_QK 0
#endif
    // Experimental: INT8 QK (Q, K int8 with per-token fp32 scales), PV stays BF16.
    static constexpr bool Int8_QK = FA3_INT8_QK && !Is_FP8;
    using ElementQK = std::conditional_t<Mixed_QK, cutlass::bfloat16_t, std::conditional_t<Int8_QK, int8_t, Element>>;
    using ElementAccumQK = std::conditional_t<Int8_QK, int32_t, ElementAccum_>;
""")
m=sub(m,"op_selector<ElementQK, ElementQK, ElementAccum, TileShape_MNK>()","op_selector<ElementQK, ElementQK, ElementAccumQK, TileShape_MNK>()",2)
m=sub(m,"    static_assert(!(!MmaPV_is_RS && Is_FP8), \"MmaPV must be RS if FP8\");\n",
"""    static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
    static_assert(!Int8_QK || (IntraWGOverlap && !PackGQA && !Has_softcap && !HasQv), "INT8 QK: only IntraWGOverlap, no PackGQA/softcap/Qv");
""")
a=m.index("        if constexpr (IntraWGOverlap) {\n            Tensor tSrS = partition_fragment_C(tiled_mma_qk")
b=m.index("        } else {  // No intra-WG overlap",a)
r=m[a:b]
r=sub(r,"Tensor tSrS = partition_fragment_C(tiled_mma_qk","Tensor tSrS_qk = partition_fragment_C(tiled_mma_qk",2)
r=sub(r,"smem_pipe_read.index()), tSrS);","smem_pipe_read.index()), tSrS_qk);",4)
r=sub(r,"\n            scoremod_premask_fn(tSrS);\n","\n            auto&& tSrS = qk_to_float(tSrS_qk, n_block);\n            scoremod_premask_fn(tSrS);\n")
r=sub(r,"                scoremod_premask_fn(tSrS);\n","                auto&& tSrS = qk_to_float(tSrS_qk, n_block);\n                scoremod_premask_fn(tSrS);\n")
setup="""        // INT8 QK: per-thread row/col coordinates of the S fragment; q scales are loaded once per m_block.
        [[maybe_unused]] auto thr_mma_qk_i8 = tiled_mma_qk.get_thread_slice(thread_idx);
        [[maybe_unused]] Tensor tScS_i8 = thr_mma_qk_i8.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK{})));
        [[maybe_unused]] Tensor tScS_i8_rc = make_tensor(tScS_i8.data(), flash::convert_layout_acc_rowcol(tScS_i8.layout()));
        static constexpr int kI8Rows = Int8_QK ? int(decltype(size<0>(tScS_i8_rc))::value) : 1;
        [[maybe_unused]] float qs_i8[kI8Rows];
        if constexpr (Int8_QK) {
            float const* qs_ptr = params.ptr_q_descale + bidh * get<1>(params.stride_q_descale) + m_block * kBlockM;
            #pragma unroll
            for (int mi = 0; mi < kI8Rows; ++mi) { qs_i8[mi] = __ldg(qs_ptr + get<0>(tScS_i8_rc(mi, _0{}))); }
        }
        auto qk_to_float = [&](auto& tSrS_qk, int const n_blk) -> decltype(auto) {
            if constexpr (!Int8_QK) {
                return static_cast<std::remove_reference_t<decltype(tSrS_qk)>&>(tSrS_qk);
            } else {
                Tensor tSrS_f = make_tensor_like<float>(tSrS_qk);
                Tensor s_rc = make_tensor(tSrS_qk.data(), flash::convert_layout_acc_rowcol(tSrS_qk.layout()));
                Tensor f_rc = make_tensor(tSrS_f.data(), flash::convert_layout_acc_rowcol(tSrS_f.layout()));
                float const* ks_ptr = params.ptr_k_descale + bidh_kv * get<1>(params.stride_k_descale) + n_offset + n_blk * kBlockN;
                #pragma unroll
                for (int ni = 0; ni < size<1>(f_rc); ++ni) {
                    float const ks = __ldg(ks_ptr + get<1>(tScS_i8_rc(_0{}, ni)));
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(f_rc); ++mi) { f_rc(mi, ni) = float(s_rc(mi, ni)) * (qs_i8[mi] * ks); }
                }
                return tSrS_f;
            }
        };

"""
m=m[:a]+setup+r+m[b:]
(H/'mainloop_fwd_sm90_tma_gmma_ws.hpp').write_text(m)
L=(H/'flash_fwd_launch_template.h').read_text()
assert 'FA3_INT8_BLOCK_N' not in L
L=sub(L,"    static constexpr int kBlockN = Arch >= 90 && FA3_MIXED_QK_BF16 && Is_FP8 ? FA3_MIXED_BLOCK_N : ",
"""#ifndef FA3_INT8_QK
#define FA3_INT8_QK 0
#endif
#ifndef FA3_INT8_BLOCK_N
#define FA3_INT8_BLOCK_N 80
#endif
    static constexpr int kBlockN = Arch >= 90 && FA3_INT8_QK && !Is_FP8 ? FA3_INT8_BLOCK_N : Arch >= 90 && FA3_MIXED_QK_BF16 && Is_FP8 ? FA3_MIXED_BLOCK_N : """)
(H/'flash_fwd_launch_template.h').write_text(L)
print('patched')
