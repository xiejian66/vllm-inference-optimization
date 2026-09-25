# 正式实验入口

从仓库根目录设置 `QWENFUSE_WORKSPACE`、`QWENFUSE_MODEL`，按 `qwenfuse/docs/REPRODUCE.md` 准备并构建 A/B/C/D。

`bash qwenfuse/benchmarks/resume_metrics_suite/run.sh --out <结果目录>` 默认仅输出计划；`--execute` 才执行。支持 `--resume --execute` 恢复。`report.py <结果目录>` 汇总结果。

微基准采用 `stable_micro.py` 的 Graph 内部计时事件；模型实验采用配对顺序和固定种子。`legacy_suite.py` 仅作为冻结/执行公共实现使用，正式入口是 `run_suite.py`。
