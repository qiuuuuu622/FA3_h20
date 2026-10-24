# 实验归档与复现

来源为 h20-2 `/root/gdn35/20261024-fp8-accuracy/`。原始脚本、结果与日志保持原样；新增 `build.py` 用于重定位原始构建命令。

## 文件导航

| 路径 | 内容 |
|---|---|
| `fa3_candidate/hopper/` | 已应用补丁的完整 Hopper 源码快照及依赖头文件 |
| `fa3_candidate/entry_int8bf.cu` | 固定形状 C ABI 入口 |
| `fa3_candidate/backup_pre_fusedq_125547/` | fused-Q 改动前 mainloop |
| `patches/fused-q.diff` | 从备份到当前 mainloop 的精确差异 |
| `scripts/33_patch_fused_q.py` | 在 pre-fused-Q mainloop 上应用增量补丁 |
| `scripts/34_int8bf_fusedq_bench.py` | capture 精度、重复性、CUDA Graph 交替顺序计时 |
| `scripts/32_int8bf_boundaries.py` | 读取 34 号脚本定义，INT8BF_FUSEDQ=1 时传入 BF16 Q |
| `scripts/35_fusedq_summary.py` | 原始统计工具 |
| `fa3_candidate/fa3_int8bf_fusedq_n{80,96}.so` | 实验机器原始二进制；仅适用兼容 CUDA/SM90a 环境 |
| `evidence/` | 原始编译命令、上游版本、SHA256 清单 |
| `results/`、`logs/` | 原始 JSONL、导出 CSV、summary、完整 sanitizer 日志 |

## 不需要 GPU 的数据复算

在本实验目录执行：

```bash
python3 scripts/35_fusedq_summary.py results/int8bf_fusedq_bench.jsonl
```

## 构建

需要 Linux/H20、支持 SM90a 的 CUDA nvcc，以及 CUTLASS commit `7127592069c2fe01b041e174ba4345ef9b279671`。上游 sgl-attn commit 记录在 `evidence/source_manifest.json`。原实验 Python/CUDA 依赖的完整版本锁定未随指定证据提供，不能保证跨环境逐位复现。

```bash
git clone https://github.com/NVIDIA/cutlass.git /tmp/fa3-cutlass
git -C /tmp/fa3-cutlass checkout 7127592069c2fe01b041e174ba4345ef9b279671
python3 scripts/build.py int8bf_fusedq_n96 --cutlass /tmp/fa3-cutlass
python3 scripts/build.py int8bf_fusedq_n80 --cutlass /tmp/fa3-cutlass
python3 scripts/build.py int8bf_n96 --cutlass /tmp/fa3-cutlass
python3 scripts/build.py int8bf_ctrl_bf16 --cutlass /tmp/fa3-cutlass
python3 scripts/build.py c128 --cutlass /tmp/fa3-cutlass
```

归档源码已应用 fused-Q 补丁，不要再次直接运行 33 号补丁。验证补丁可将备份 mainloop 复制到临时目录后运行 `python3 scripts/33_patch_fused_q.py 临时目录`，并与当前 mainloop 比较。基础 29 号补丁是历史参考，也不要重复应用。

## 原数据 GPU 复测

需要 torch、scipy、含 `sgl_kernel.flash_attn` 的兼容环境，以及原始 `captures/sample_*_layer*.pt`。捕获张量未纳入 Git，因此仅凭本仓库不能重新产生同一批精度结果。每个 capture 包含 q、k_cache、v_cache、output 及 meta.kv_tokens；不能用随机数据替代来声称复现原精度表。

将原 captures 放入本实验目录后，可执行：

```bash
python3 scripts/34_int8bf_fusedq_bench.py --root "$PWD" \
  --variants fa3_int8bf_n96.so,fa3_int8bf_fusedq_n96.so,fa3_int8bf_fusedq_n80.so \
  --timing --out fusedq_rerun.jsonl
```

结果以 append 模式写入，复测请用新文件名。默认 reps=5、rounds=6；CUDA Graph 预热后交替正序/倒序测量。不能从 summary 单独确认原始启动参数，以 JSONL 中实际 times_ms 为准。

边界脚本保留原始默认绝对路径 `/root/gdn35/20261024-fp8-accuracy`，不支持 `--root`（会重写 argv）。在原机器、原目录与兼容环境执行：

```bash
INT8BF_LIB=fa3_int8bf_fusedq_n96.so INT8BF_FUSEDQ=1 \
  compute-sanitizer --tool memcheck python3 scripts/32_int8bf_boundaries.py
```

其他检查分别替换 tool 为 initcheck、racecheck、synccheck。迁移路径时需先修改 34 号脚本的 `--root` 默认值，或为原绝对路径提供相同目录布局。已存 synccheck 是失败证据，详见 RESULTS.md。
