# 计算环境（checklist 4.8）

## 论文主实验应记录的硬件

当前主机于 2026-07-28 实测：

| 项目 | 配置 |
|---|---|
| GPU 型号 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 实际可见 GPU 数量 | 8 |
| 单卡显存 | 97,887 MiB（约 96 GiB） |
| CPU | 2 × Intel Xeon Platinum 8457C |
| CPU 逻辑核 | 192 |
| 系统内存 | 1.0 TiB |
| 当前数据盘 | 3.5 TiB，总剩余约 426 GiB |
| 代码磁盘需求 | 小于 1 MiB（不含虚拟环境） |
| 建议实验磁盘需求 | 至少 250 GiB；模型、页面数据、FAISS 索引和输出另计 |

该机器是多人共享服务器；检查时多个 GPU 上存在其他用户/任务进程。实验没有独占
整机，也不应声称独占全部 8 张 GPU。“实际使用 GPU 数量”必须按每次正式运行填写，不能仅写机器可见数量。建议在结果
目录保存 `CUDA_VISIBLE_DEVICES`、`nvidia-smi -L` 和运行命令。单个 embedding
模型通常使用 1 GPU；大生成模型的实际张量并行卡数由 vLLM 启动参数决定。

建议逐实验记录：

| 实验 | 实际/计划 GPU 数量 | 说明 |
|---|---:|---|
| 文本 embedding 索引与检索 | 1 | 单模型串行 |
| 多模态 embedding 索引与检索 | 1 | 单模型串行 |
| M3DocRAG / MDocAgent / ViDoRAG 检索 | 1 | 两路模型可顺序加载 |
| HippoRAG | 2 个服务 GPU（LLM + embedding） | 客户端本身可在 CPU |
| 8B/32B 生成 | 1（96 GiB 卡，视精度而定） | 必须保存实际 vLLM 参数 |
| 235B 生成 | 最多 8 | 以实际 tensor-parallel-size 为准 |

## 软件

当前主机基础环境实测：

| 项目 | 版本 |
|---|---|
| 操作系统 | Ubuntu 22.04 系列，Linux 6.8.0-114-generic |
| Python | 3.12.11 |
| NVIDIA Driver | 580.126.20 |
| 驱动支持的 CUDA | 13.0 |
| CUDA Toolkit (`nvcc`) | 12.8 |
| cuDNN | 当前主机系统包无法确认；Docker 目标为 CUDA 12.4 cuDNN-devel |

独立复现环境的关键依赖记录在 `requirements.txt`。当前基础 Python 环境没有安装
下列包，因而不能把版本误写成主实验实际版本：

| 依赖 | 复现约束 / 状态 |
|---|---|
| PyTorch | 主环境目标 2.5.x；当前基础环境未安装，正式运行后记录实际版本 |
| Transformers | 主环境约束 `>=4.51`；当前基础环境未安装 |
| vLLM | HippoRAG 参考依赖 0.6.6.post1；生成服务端必须另存实际版本和启动命令 |
| FAISS | `faiss-cpu>=1.8`；GPU 索引时记录对应 GPU 包版本 |
| MinerU | 页面解析阶段依赖；历史实际版本未保留，发布前必须从原数据制备环境补录 |
| Sentence Transformers | `>=3.0` |
| OpenAI Python SDK | `>=1.40`，用于兼容 OpenAI/vLLM API |
| ColPali Engine | `>=0.3`，用于 M3DocRAG/ViDoRAG 多向量检索 |
| scikit-learn | `>=1.5`，用于 ViDoRAG GMM 候选筛选 |
| HippoRAG | 参考实现为 `2.0.0-alpha.4`；建议独立 Python 3.10 环境 |

## 正式运行时生成环境快照

```bash
python --version
uname -a
nvidia-smi
nvcc --version
python -m pip freeze > "${OUTPUT_DIR}/pip-freeze.txt"
python - <<'PY' > "${OUTPUT_DIR}/key-versions.txt"
from importlib.metadata import version, PackageNotFoundError
for name in ("torch", "transformers", "vllm", "faiss-cpu", "mineru",
             "sentence-transformers", "openai"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "not installed")
PY
```

上述快照应与实验输出一同归档，论文中填写的是实际运行环境，而不是
`requirements.txt` 的最低版本。
