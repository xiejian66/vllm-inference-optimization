# 指标与实验口径

## 算子消融

A4 = QK Norm + RoPE，再独立 KV 写入；B4 = 纳入 KV 写入融合；C4 = B4 加多 head 路径 cos/sin 寄存器预加载。三组固定 heads/warp=4、相同形状、dtype 和布局。

正式展示的稳定计时数据来自：
- `results/targeted-diag-20260925-141923/micro/summary.txt`：tokens=8、128。
- `results/micro-extra-20260925-145713/results/summary.txt`：tokens=1、32、64、256。

每个形状取各版本三轮中位数，再计算 `(A-B)/A`、`(B-C)/B`、`(A-C)/A`。这是整段算子路径耗时，不能把两项降幅相加，也不代表模型端到端等幅提升。Graph 捕获输入 reset 在计时区间之外，计时开始/结束事件均位于 Graph 内。268 项既有算子比较与模型 logits 比较是两个不同层次。

## 调度与整体

历史汇总：`results/resume-metrics-20260924-225223/summary.txt`。
S0=官方 FCFS+APC；S1=默认算子+风险感知准入；JOINT=C4+风险感知准入。APC 保持开启；三个随机种子，两种顺序，固定输入输出、KV 预算及请求组成。

分别汇报整批耗时、输出 tok/s、冷热 TTFT/TPOT、缓存命中和抢占。吞吐增幅=`ON/OFF-1`，延迟降幅=`1-ON/OFF`。六组百分比的中位数不等于所有样本混在一起的比值。

S1 的一组负收益、冷请求等待代价、16 GiB 和全冷对照均保留；不只展示最有利场景。历史汇总前半部分的旧微基准存在漂移，算子展示数字使用上面的稳定补测。

## 限制

少量相同历史位置上的 argmax 一致不代表完整模型精度无损。保留 numerical-review 与 comparison 记录。不报告未测的 P99、显存容量压缩或持续线上服务收益。发布目录此次仅进行了 CPU 级工具验证，没有独立构建和 GPU 复测。
