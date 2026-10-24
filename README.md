# FA3 on H20：INT8 QK / BF16 PV 与融合 Q 量化

归档 2026-10-24 H20 实验的核心 CUDA 源码、补丁、编译命令、测试代码和原始结果。

- [对比表与精度、sanitizer 结果](experiments/20261024-fused-q/RESULTS.md)
- [优化机制](experiments/20261024-fused-q/MECHANISM.md)
- [构建与复现说明](experiments/20261024-fused-q/README.md)
- [核心 fused-Q diff](experiments/20261024-fused-q/patches/fused-q.diff)

融合 Q 后，n96 相对单独测量的预量化 attention + Q 预处理合计节省约 5.11% 时间；相对 BF16 的逐样本平均加速约 1.2617×。当前仍有严格精度检查失败和 synccheck 同步错误，属于实验实现。

上游 sgl-attn 源码及其修改保留 [BSD-3-Clause 许可证](experiments/20261024-fused-q/fa3_candidate/LICENSE)。版本信息见实验目录 `evidence/source_manifest.json`。
