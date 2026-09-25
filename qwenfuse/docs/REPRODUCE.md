# 复现指南

所有命令从仓库根目录运行。安装、获取源码、编译与运行实验是独立步骤；没有启动即自动长时间执行的操作。

## 1. 环境

已测：A100 40GB (SM80)，Python 3.12，PyTorch 2.13.0+cu130，CUDA Toolkit 13.0，FlashInfer 0.6.18.post1。参考 `environment-main.freeze.txt`，其中旧 editable 安装路径仅为历史记录，不应直接用于 pip 安装。

创建或激活自己的环境，确认 `python`、`nvcc`、`g++` 可用。以下为后续安装命令，本次整理未执行它们：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"
# Torch CUDA wheel 使用独立索引，可按网络情况替换。
python -m pip install 'torch==2.13.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/build/cuda.txt -r requirements/cuda.txt
python -m pip install pytest pytest-mock
```

固定旧基线的某些依赖可能需要指定版本的 wheel 源；安装失败应检查来源与版本，不要直接升级源码基线。CUDA Toolkit 必须与构建需求匹配，驱动由宿主机提供。

## 2. 工作目录与模型

```bash
export QWENFUSE_WORKSPACE="$PWD/.qwenfuse-work"
export QWENFUSE_MODEL="/absolute/path/to/Qwen3-8B"
```

工作目录存放消融源码、构建产物、实验输出，不提交 Git。模型目录应包含配置和完整权重。可通过 `QWENFUSE_REPO_A/B/C/D` 分别覆盖消融源码路径。

## 3. 准备消融版本（不编译）

```bash
python qwenfuse/scripts/prepare_variants.py                 # 只展示计划
python qwenfuse/scripts/prepare_variants.py --execute       # 获取固定基线并应用补丁
```

也可使用已有、包含基线 commit 的本地 Git 仓库：

```bash
python qwenfuse/scripts/prepare_variants.py \
  --base-repo /absolute/path/to/local-vllm --execute
```

脚本在 workspace 中创建独立 Git 仓库和 A/B/C/D 工作树，拒绝覆盖已有目标。A 是固定官方基线；B/C/D 分别应用 `base-to-B/C/D.patch`，补丁不能串行叠加。D 是实验 overlay 的调度输入版本，最终风险钩子由实验工具接入；发布根目录已预先接入这些钩子。

## 4. 构建（后续手动执行，本次未验证）

激活上述环境并使用 GPU 对应的架构。A100 示例：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
(
  set -e
  for variant in A B C; do
    (cd "$QWENFUSE_WORKSPACE/src/$variant" && python setup.py build_ext --inplace)
  done
)
```

每个版本需有 `vllm/_C_stable_libtorch.abi3.so`，A 还需具备 FA2 等辅助扩展。实验前检查实际构建是否成功，不能仅凭循环结束判断成功。

D 的 Python 调度代码不需要另一份独立 CUDA 编译，可以显式链接 C 主扩展和 A 辅助扩展：

```bash
python qwenfuse/scripts/link_extensions.py \
  --main "$QWENFUSE_WORKSPACE/src/C" --aux "$QWENFUSE_WORKSPACE/src/A" \
  --target "$QWENFUSE_WORKSPACE/src/D" --execute
```

运行期依赖必须已安装；源码构建不等于运行依赖已全部满足。完整构建行为以该基线的构建脚本为准。

## 5. 检查与正式实验

```bash
PYTHONDONTWRITEBYTECODE=1 python qwenfuse/scripts/check_package.py
bash qwenfuse/benchmarks/resume_metrics_suite/run.sh \
  --out "$QWENFUSE_WORKSPACE/reports/run-01" --preflight
```

preflight 会检查模型、GPU 和编译扩展，但不加载模型；它不是推理验收。确认后启动：

```bash
bash qwenfuse/benchmarks/resume_metrics_suite/run.sh \
  --out "$QWENFUSE_WORKSPACE/reports/run-01" --execute
# 中断后的恢复使用同一结果目录和 --resume --execute。
```

总计划包含 94 个任务。微基准现在使用稳定的 Graph 内部事件计时，128 次独立输入调用、100 次采样、3 秒预热；与历史早期微基准结果不能混算。模型实验设置保持原计划。

## 6. 分析

```bash
python qwenfuse/benchmarks/resume_metrics_suite/report.py \
  "$QWENFUSE_WORKSPACE/reports/run-01"
```

进一步诊断需完整运行目录：

```bash
python qwenfuse/benchmarks/diagnose_metrics.py micro \
  --snapshot "$QWENFUSE_WORKSPACE/reports/run-01" \
  --out "$QWENFUSE_WORKSPACE/reports/micro-review-01"
```

仓库内 `results/` 是精简证据，不含 overlays、动态库和完整 tensor 数据，不能当作 `--snapshot` 的输入。调度诊断脚本使用同一组环境变量；历史 slot-guard 对照另需通过 `QWENFUSE_SLOT_GUARD_BEFORE` 指定修改前 scheduler 文件，它不属于正式复现入口。
