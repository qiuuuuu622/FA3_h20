// INT8 QK / BF16 PV entry (q,k int8 + per-token fp32 scales; v bf16). See scripts/29_patch_int8qk_bf16pv.py.
// qs: [16, nq_pad] (head stride nq_pad), ks: [2, nk_pad] (head stride nk_pad); both padded with zeros.
#include <cuda_runtime.h>
#include "flash_fwd_launch_template.h"
void prepare_varlen_num_blocks(Flash_fwd_params &, cudaStream_t, bool, int, int, bool) {
  fprintf(stderr,"unexpected scheduler prepare in single-tile specialization\n"); abort();
}
extern "C" int int8bf_run(void* q, void* k, void* v, void* out, void* lse,
  float* qs, float* ks, int* pages, int* semaphore, int nq, int nk, int nq_pad, int nk_pad, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  Flash_fwd_params p{};
  p.q_ptr=q; p.k_ptr=k; p.v_ptr=v; p.o_ptr=out; p.softmax_lse_ptr=lse;
  p.q_row_stride=4096; p.o_row_stride=4096; p.k_row_stride=p.v_row_stride=512;
  p.q_head_stride=p.o_head_stride=p.k_head_stride=p.v_head_stride=256;
  p.q_batch_stride=p.o_batch_stride=(int64_t)nq*4096;
  p.k_batch_stride=p.v_batch_stride=8192; p.v_dim_stride=1;
  p.b=p.b_k=1; p.h=16;p.h_k=2;p.d=p.dv=p.d_rounded=p.dv_rounded=256;
  p.seqlen_q=p.total_q=nq;p.seqlen_k=(nk+15)/16*16;p.total_k=nk;
  p.seqused_k=semaphore+1;p.skip_scheduler_metadata_computation=true;
  p.seqlen_q_rounded=(nq+127)/128*128;p.seqlen_k_rounded=(nk+127)/128*128;
  p.scale_softmax=1.f/16;p.is_e4m3=false;p.is_bf16=true;p.is_causal=true;
  p.q_descale_ptr=qs;p.k_descale_ptr=ks;p.v_descale_ptr=nullptr;
  p.q_descale_batch_stride=p.k_descale_batch_stride=p.v_descale_batch_stride=0;
  p.q_descale_head_stride=nq_pad;p.k_descale_head_stride=nk_pad;p.v_descale_head_stride=0;
  p.page_table=pages;p.page_size=16;p.num_pages=(nk+15)/16;p.page_table_batch_stride=p.num_pages;
  p.window_size_left=-1;p.window_size_right=0;p.num_splits=1;p.arch=90;p.num_sm=78;
  p.tile_count_semaphore=semaphore;
  CHECK_CUDA(cudaMemsetAsync(semaphore,0,sizeof(int),stream));
  run_flash_fwd<90,256,256,1,cutlass::bfloat16_t,cutlass::bfloat16_t,
    true,false,false,true,true,false,false,false,false,false,false,false>(p,stream);
  return (int)cudaGetLastError();
}
