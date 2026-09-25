# vLLM Inference Optimization：基于 vLLM 的算子与 KV Cache 调度优化

面向 Qwen3-8B BF16 推理，从 Attention 前处理和请求准入两部分改进执行效率：融合 QK RMSNorm、RoPE 与 KV Cache 写入，在多 head/warp 分支预加载 cos/sin，并根据潜在前缀缓存淘汰风险调整等待队列的准入顺序。

**实验平台：NVIDIA A100-PCIE-40GB，固定 vLLM 基线 `27757dde020ecda4f9b0e2c5ca2df1c29badc82f`。** 完整源码位于仓库根目录；实验工具与结果位于 `qwenfuse/`。

## 设计

- **Attention 前处理融合**：在已有 QK-Norm + RoPE 路径中纳入 KV 写入，减少一次独立 kernel 调用及中间数据往返。源文件：`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`。
- **寄存器预加载**：在多 head/warp 路径中调整 cos/sin 加载方式。消融实验固定 heads/warp=4，以区分 KV 写入融合与寄存器预加载的收益。
- **风险感知准入**：存在运行名额时，根据候选请求完整 prompt 的块需求检查空闲队列中的缓存暴露风险；触发后在有限窗口内优先复用前缀。更换候选后重新计算命中和分配需求，限制重排重试。该策略不保证所有热前缀都能保住，也不保证冷请求延迟总是下降。

## 实测结果

下列是已经完成的历史实验，整理发布目录后未重新执行 GPU 实验。

| 对照 | 条件 | 结果 |
|---|---|---|
| A4 → B4：增加 KV 写入融合 | CUDA Graph，4 heads/warp，tokens=1/8/32/64/128/256 | 各形状三轮中位数：路径耗时下降约 14.18%～27.54% |
| B4 → C4：再加寄存器预加载 | 同上 | 路径耗时进一步下降约 11.70%～16.85% |
| A4 → C4：两步合计 | 同上 | 路径耗时下降约 28.64%～36.96% |
| S0 → S1：只改准入策略 | 8 GiB KV 预算、随机冷热混合批次 | 六组配对吞吐增幅中位数约 5.02%，5/6 组为正 |
| S0 → JOINT：算子与调度组合 | 同一随机混合负载 | 六组配对吞吐增幅中位数约 6.24%，6/6 组为正 |

调度负载含 64 个请求，每个输入 2,176 tokens、固定输出 256 tokens，32 个冷请求与 32 个带可复用前缀的请求随机打乱，最大活动请求数为 16。采用三个种子和两种执行顺序；这些是整批突发负载结果，不是持续服务吞吐或 P99 指标。

S0/S1 使用官方原生算子及默认 warp 选择；JOINT 使用 C4 算子，因此 S1→JOINT 不等于纯粹的寄存器预加载收益。冷请求 TTFT 在部分负载中上升；16 GiB 与全冷对照没有稳定收益。详细口径与原始摘要见 [实验说明](qwenfuse/docs/EXPERIMENTS.md)。

## 环境与快速入口

已用环境：Linux x86_64、Python 3.12、PyTorch 2.13.0+cu130、CUDA Toolkit 13.0、FlashAttention 2。模型与编译扩展不包含在仓库中。

```bash
# 在仓库根目录执行；先激活自己的 Python 环境。
export QWENFUSE_WORKSPACE="$PWD/.qwenfuse-work"
export QWENFUSE_MODEL="/absolute/path/to/Qwen3-8B"

# 仅检查代码、测试计划与汇总逻辑，不加载模型或编译。
PYTHONDONTWRITEBYTECODE=1 python qwenfuse/scripts/check_package.py

# 查看消融源码准备计划，不访问网络、不创建工作树。
python qwenfuse/scripts/prepare_variants.py

# 查看完整实验计划，不执行实验。
bash qwenfuse/benchmarks/resume_metrics_suite/run.sh \
  --out "$QWENFUSE_WORKSPACE/reports/run-01"
```

依赖安装、源码准备、构建、启动与结果分析见 [复现指南](qwenfuse/docs/REPRODUCE.md)。本发布副本已接入风险策略钩子；使用 `vllm.v1.core.sched.qwenfuse_risk.RiskPerfScheduler` 显式选择最终调度类。消融工具会独立构造 S0/S1/JOINT 的 Python 与算子组合。

## 目录

```text
vllm/                         Python 模型执行、编译和调度实现
csrc/                         CUDA/C++ 算子
tests/                       上游与新增回归测试
qwenfuse/
  scripts/                    源码准备、动态库连接和 CPU 检查
  benchmarks/resume_metrics_suite/  正式实验入口
  benchmarks/admission_probe/       调度机制诊断
  patches/                    相对固定基线的 B/C/D 消融补丁
  results/                    精选历史数据
  docs/                       环境、版本及复现说明
```

## 验证范围

算子在已有测试覆盖范围内通过逐元素比较；模型 logits 的跨版本差异仍保留记录，不能以少量 argmax 一致代替全面精度验证。本策略目前限定同步、单组 Full Attention，不覆盖所有模型、分布式配置或 KV connector。发布目录的独立构建和最小推理尚未执行。

原框架说明见 [README.upstream.md](README.upstream.md)，许可证见 [LICENSE](LICENSE)。
