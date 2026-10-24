# 2026-10-24 H20 Fused-Q 实验结果

基于原始 JSONL 的 40 条 capture 记录；每条记录先取重复计时中位数，再对 capture 等权平均。不是模型端到端推理测试。

| 方案 | 平均耗时 ms | 相对 BF16 加速¹ | 平均相对 L2 | 最差相对 L2 | 严格通过 |
|---|---:|---:|---:|---:|---:|
| `fa3_int8bf_n96` | 6.8372 | 1.2922× | 0.5108% | 0.7130% | 0/40 |
| `fa3_int8bf_fusedq_n96` | 6.9955 | 1.2617× | 0.5108% | 0.7130% | 0/40 |
| `fa3_int8bf_fusedq_n80` | 7.0427 | 1.2532× | 0.5108% | 0.7131% | 0/40 |
| `ctrl_bf16` | 9.1678 | 0.9640× | 0.0000% | 0.0000% | 40/40 |
| `bf16` | 8.8501 | 1.0000× | 0.0000% | 0.0000% | 40/40 |
| `fp8_promote1_c128` | 4.6828 | 1.8869× | 3.2059% | 6.2576% | 0/40 |

¹ 加速列是逐 capture 的 BF16/variant 比值的算术平均，不是两列平均耗时相除。

## Q 路径对比

| 路径 | 平均耗时 ms |
|---|---:|
| 预量化 n96 attention | 6.8372 |
| 单独 Torch Q Hadamard + quant | 0.5349 |
| 两者算术相加 | 7.3721 |
| fused-Q n96 attention（包含 Q 转换） | 6.9955 |

相对分开测量后的合计，融合减少 5.11% 耗时，约 1.0538×；相对已经准备好 INT8 Q 的 attention 内核则慢 2.32%。合计仅为估算，并非同一执行图的联合测量。`q_prep` 只运行 `q8_tok(q.float() @ H)`，没有完整 `prep()` 中的 INT8 cast、contiguous 和 scale 转置写入。所有方案都排除 K 预处理、模型其他算子及服务开销。

## 精度与验证边界

严格通过定义为 `torch.allclose(atol=0.003, rtol=0.01)`。三个 INT8 方案均为 0/40，不能宣称满足该精度要求；平均逐元素通过比例约 98.67%。输出有限、三次重复逐位一致均为 40/40；这不等于数值正确或同步安全证明。Fused-Q 与量化模拟平均相对 L2 约 0.1687%。

| 已保存 n96 sanitizer 检查 | 结果 |
|---|---|
| memcheck | 0 errors |
| initcheck | 0 errors |
| racecheck | 0 hazards / errors / warnings |
| synccheck | **失败：Missing wait；1824 errors；unspecified launch failure** |

边界覆盖 `(nq,nk)`：`(1,1), (1,17), (15,31), (16,32), (17,49), (63,129), (65,513), (129,2049)`。synccheck 根因尚未确认，不能据其他检查通过判定是工具误报。n80 没有随本次提供的 sanitizer 证据。

原始证据：`results/int8bf_fusedq_bench.jsonl`、`logs/int8bf_fusedq_summary.log`、`logs/sanitize_*`。本次整理仅复算和核对已有数据，没有重新运行 GPU 基准或编译。
