# AgentInferLab 环境核验报告

- 核验日期：2026-07-16（Asia/Shanghai）
- 项目目录（Windows）：`D:\agent-infer-lab`
- 目标运行环境：WSL2 / Ubuntu 24.04.3 LTS
- 最终状态：**通过**

## 1. 验收结论

| 验收项 | 状态 | 核验结果 |
|---|---:|---|
| Linux / WSL2 | 通过 | WSL 2.7.10.0；Ubuntu 24.04.3 LTS；Linux 6.18.33.2 |
| WSL 存储位置 | 通过 | Ubuntu `BasePath` 为 `D:\WSL\Ubuntu`，默认用户 UID 1000；C 盘旧 VHD 不存在 |
| NVIDIA GPU | 通过 | NVIDIA GeForce RTX 4060，8188 MiB，Compute Capability 8.9 |
| 驱动与 GPU 透传 | 通过 | Windows 驱动 581.57；WSL 可访问 GPU |
| Linux CUDA Toolkit | 通过 | CUDA Toolkit 12.9.2；`nvcc` V12.9.86 |
| Python 虚拟环境 | 通过 | Python 3.12.3；`/home/ayax/.venvs/agent-infer-lab` |
| PyTorch CUDA | 通过 | PyTorch 2.11.0+cu129；`torch.cuda.is_available()` 为 `True` |
| GPU 实算 | 通过 | RTX 4060 上计算平方张量，结果为 `[1.0, 4.0, 9.0, 16.0, 25.0]` |
| vLLM | 通过 | vLLM 0.23.0+cu129；CLI 和 Python 导入均成功 |
| Python 依赖 | 通过 | `pip check` 输出 `No broken requirements found.` |
| Nsight Systems | 通过 | Linux CLI 2025.1.3 |
| Nsight Compute | 通过 | Linux CLI 2025.2.1 |
| Debian 软件包 | 通过 | `dpkg --audit` 无异常输出 |

## 2. 版本与路径

### Windows 宿主机

| 项目 | 核验值 |
|---|---|
| Windows | Microsoft Windows NT 10.0.19045.0 |
| PowerShell | 5.1.19041.5737 |
| GPU | NVIDIA GeForce RTX 4060 |
| 显存 | 8188 MiB |
| Compute Capability | 8.9 |
| NVIDIA 驱动 | 581.57 |
| `nvidia-smi` 显示的 CUDA Version | 13.0（驱动可支持的最高运行时，不等于已安装 Toolkit） |
| Windows CUDA Toolkit | 12.9，`nvcc` V12.9.41 |
| Windows `nvcc` | `D:\CUDA12.9\bin\nvcc.exe` |
| Nsight Systems | 2025.6.1 |
| Nsight Compute | 2025.2.0 |

### WSL2 / Linux

| 项目 | 核验值 |
|---|---|
| WSL | 2.7.10.0 |
| Ubuntu | 24.04.3 LTS (Noble Numbat) |
| Linux 内核 | 6.18.33.2-microsoft-standard-WSL2 |
| Linux 用户 | `ayax` |
| Ubuntu VHD | `D:\WSL\Ubuntu\ext4.vhdx`，约 39.23 GiB |
| CUDA Toolkit | 12.9.2，`/usr/local/cuda-12.9` |
| `nvcc` | 12.9.86，`/usr/local/bin/nvcc` |
| `nsys` | 2025.1.3，`/usr/local/bin/nsys` |
| `ncu` | 2025.2.1，`/usr/local/bin/ncu` |
| Python | 3.12.3 |
| Python 虚拟环境 | `/home/ayax/.venvs/agent-infer-lab`，约 14 GiB |
| CPU 开发环境 | `/home/ayax/.venvs/agent-infer-lab-dev` |
| PyTorch | 2.11.0+cu129 |
| PyTorch CUDA | 12.9 |
| vLLM | 0.23.0+cu129 |
| uv | 0.11.29，`/home/ayax/.local/bin/uv` |
| pytest | 9.1.1，仅安装在 CPU 开发环境 |
| Ruff | 0.15.22，仅安装在 CPU 开发环境 |
| vLLM 官方 wheel 缓存 | `D:\AIInfra\cache\vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` |
| wheel SHA256 | `8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1` |
| pip 缓存 | `/home/ayax/.cache/pip`，约 6.2 GiB |
| uv 缓存 | `/home/ayax/.cache/uv` |

所有 Linux 环境数据都位于 Ubuntu 的 D 盘 VHD 中；项目源码位于 `D:\agent-infer-lab`，在 WSL 中对应 `/mnt/d/agent-infer-lab`。

## 3. 关键验收证据

```text
$ nvcc --version
Cuda compilation tools, release 12.9, V12.9.86

$ nsys --version
NVIDIA Nsight Systems version 2025.1.3...

$ ncu --version
Version 2025.2.1.0

$ python -m pip check
No broken requirements found.

$ vllm --version
0.23.0+cu129
```

GPU 实算的核心输出：

```text
torch=2.11.0+cu129
torch_cuda=12.9
cuda_available=True
device=NVIDIA GeForce RTX 4060
gpu_result=[1.0, 4.0, 9.0, 16.0, 25.0]
vllm=0.23.0
```

## 4. 版本决策说明

当前 vLLM 的 PyPI 默认 wheel 已切换到 CUDA 13，直接安装得到的 vLLM 0.25.1 依赖 `libcudart.so.13`，与本项目选定的 PyTorch/CUDA 12.9 栈不一致。最终改用 vLLM 官方 GitHub Release 的 `0.23.0+cu129` 精确 wheel，并核对 GitHub 资产大小与 SHA256 后安装。

采用 CUDA 12.9 的原因：

1. Windows 与 Linux 开发工具链统一为 CUDA 12.9。
2. PyTorch 2.11.0+cu129 与 vLLM 0.23.0+cu129 的二进制版本一致。
3. 避免同时维护 CUDA 12.9 与 13.0 两套系统 Toolkit，减少 ABI 和排错变量。

参考：

- [vLLM GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM v0.23.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 5. 使用方式

```bash
source /home/ayax/.venvs/agent-infer-lab/bin/activate
cd /mnt/d/agent-infer-lab

python --version
nvcc --version
vllm --version
```

CPU 开发与质量检查：

```bash
export UV_PROJECT_ENVIRONMENT=/home/ayax/.venvs/agent-infer-lab-dev
export UV_CACHE_DIR=/home/ayax/.cache/uv

/home/ayax/.local/bin/uv sync --locked
/home/ayax/.local/bin/uv run --locked pytest -q
/home/ayax/.local/bin/uv run --locked ruff check .
/home/ayax/.local/bin/uv run --locked python scripts/check_environment.py --json
```

## 6. 已知限制与下一阶段

- vLLM 在 WSL 中提示 `pin_memory=False`，可能降低部分数据传输性能。后续基准测试必须保留该条件，不能把 WSL 结果直接等同于原生 Linux 服务器。
- 当前只完成环境与 GPU 实算验收，尚未下载模型，也未完成端到端 vLLM 推理服务验收。
- 下一阶段应先固化依赖清单和一键验收脚本，再选择适合 8 GiB 显存的小模型做最小推理闭环。
