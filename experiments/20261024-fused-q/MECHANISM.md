# 优化机制

## INT8 QK + BF16 PV 基础路径

Q、K 在 256 维 head 维度做归一化 Hadamard 变换 H/16；同一正交变换在精确算术下保持 QK 内积。K 再沿 token 维度减均值：其引入的每行 logit 常数在精确 softmax 中抵消，但量化和有限精度仍会产生误差。

Q 使用逐 token/head absmax scale；K 按 `g=(t//128)*4+(t%8)//2` 分组取 absmax。量化范围 [-127,127]。QK 使用 INT8 GMMA、INT32 累加后还原 scale，PV 保留 BF16，避免将 V 压到 FP8。实现入口为 `entry_int8bf.cu`，基础补丁为 `29_patch_int8qk_bf16pv.py`。

## Fused-Q 路径

`FA3_INT8_QK=1` 与 `FA3_INT8_QK_FUSED_Q=1` 联合启用；宏默认关闭。Q 指针改为 BF16，K 仍是预量化 INT8。

1. Consumer warp 每个负责 16 行 Q，每 lane 用 uint4 读取 8 个 BF16，总计每行 256 个元素；越界行以零填充。
2. FP32 寄存器内执行 3 级线程内蝶形和 5 级 `shfl_xor`，乘 1/16 完成 FWHT。
3. warp 归约 absmax，计算 `max(amax,1e-10)/127`，round-to-nearest 后 clamp 到 [-127,127]。
4. INT8 Q 写入 swizzled shared memory，scale 根据 S fragment 的行坐标直接写入 `qs_i8` 寄存器。
5. Producer 跳过 Q TMA prefetch 和 load；consumer 跳过原 Q barrier wait，使用 `fence_view_async_shared()` 与 `QueryRotated` named barrier，使共享内存写入对 GMMA 可见。

收益来自消除独立 Q 转换路径及其 INT8 Q/scale 中间落地。代价是 consumer 承担转换、归约和同步，因此融合 attention 本身比预量化 attention 慢。现有数据仅支持 Q 路径时间节省，不能外推服务端到端吞吐收益。

## 专用范围与未解决问题

入口固定 H20/SM90a、78 SM、B=1、Hq=16、Hkv=2、D=256、page=16、causal；不是通用 FA3 替代库。静态断言限制 head dimension 与 block/warp 行映射。n96 与 n80 为 BlockN 取值。

synccheck 检测到 Missing wait 并导致 launch failure，需进一步调查 barrier 生命周期；现阶段不能声称上述同步实现已正确。精度误差和性能数字详见 RESULTS.md。
