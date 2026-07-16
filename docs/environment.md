# AgentInferLab 环境核验报告

- 核验日期：2026-07-16（Asia/Shanghai）
- 项目目录（Windows）：`D:\agent-infer-lab`
- 目标运行环境：WSL2 / Ubuntu
- 核验方式：仅执行只读命令；本次未安装、升级或卸载任何软件

## 1. 验收结论

| 验收项 | 状态 | 结论 |
|---|---:|---|
| Linux / WSL2 | 通过 | 默认发行版为 Ubuntu 24.04.3 LTS，运行在 WSL2 |
| NVIDIA GPU 与显存 | 通过 | NVIDIA GeForce RTX 4060，8188 MiB，Compute Capability 8.9 |
| GPU 驱动与 WSL GPU 透传 | 通过 | Windows 驱动 581.57；WSL 可通过 `/usr/lib/wsl/lib/nvidia-smi` 访问 GPU |
| CUDA Toolkit | 部分通过 | Windows 已安装 CUDA Toolkit 12.9；WSL 未安装 Linux CUDA Toolkit |
| WSL `nvcc` | **未通过** | WSL 中找不到 Linux `nvcc` |
| WSL Python | 通过 | Python 3.12.3，路径 `/usr/bin/python3` |
| WSL PyTorch CUDA | **未通过** | WSL 未安装 `pip` 与 `torch`，暂时无法得到 `torch.cuda.is_available() == True` |
| Nsight Systems | 部分通过 | Windows 已安装 2025.6.1；WSL 中无 `nsys` |
| Nsight Compute | 部分通过 | Windows 已安装 2025.2.0；WSL 中无 `ncu` |

**当前阶段未达到完成条件。** GPU 与 WSL2 基础能力正常，但 Linux CUDA 编译工具链、WSL Python 环境和 Linux Nsight CLI 尚未安装。

## 2. Windows 宿主机

| 项目 | 核验值 |
|---|---|
| Windows | Microsoft Windows NT 10.0.19045.0 |
| PowerShell | 5.1.19041.5737 |
| GPU | NVIDIA GeForce RTX 4060 |
| 显存 | 8188 MiB |
| Compute Capability | 8.9 |
| NVIDIA 驱动 | 581.57 |
| `nvidia-smi` 显示的 CUDA Version | 13.0 |
| Windows CUDA Toolkit | 12.9，`nvcc` V12.9.41 |
| Windows `nvcc` | `D:\CUDA12.9\bin\nvcc.exe` |
| Windows Python | 3.13.11，`C:\Users\BayMax\AppData\Local\Programs\Python\Python313\python.exe` |
| Nsight Systems | 2025.6.1 |
| Nsight Compute | 2025.2.0 |

注意：`nvidia-smi` 中的 CUDA 13.0 表示驱动能够支持的最高 CUDA 运行时版本，不代表 WSL 已安装 CUDA Toolkit 13.0。实际发现的 Windows Toolkit 为 12.9。

## 3. WSL2 / Linux

| 项目 | 核验值 |
|---|---|
| WSL | 2.7.10.0 |
| 默认发行版 | Ubuntu，WSL Version 2 |
| Linux 内核 | 6.18.33.2-microsoft-standard-WSL2 |
| Ubuntu | 24.04.3 LTS (Noble Numbat) |
| Linux 用户 | `ayax` |
| Home | `/home/ayax` |
| GPU 访问 | 通过 `/usr/lib/wsl/lib/nvidia-smi` 成功 |
| GPU | NVIDIA GeForce RTX 4060，8188 MiB |
| 驱动 | 581.57 |
| Python | 3.12.3，`/usr/bin/python3` |
| `pip` | 未安装 |
| PyTorch | 未安装 |
| Linux CUDA Toolkit | 未安装；未发现 `/usr/local/cuda*` 或 `/opt/cuda*` |
| Linux `nvcc` | 不可用 |
| Linux `nsys` | 不可用 |
| Linux `ncu` | 不可用 |

Windows 的 `nvcc.exe` 即使可以从 WSL 路径访问，生成的也是 Windows 目标程序，不能替代项目所需的 Linux `nvcc`。

## 4. 关键核验命令

```powershell
wsl.exe --version
wsl.exe --status
wsl.exe --list --verbose
nvidia-smi.exe --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
nvcc.exe --version
nsys.exe --version
```

```bash
uname -srmo
cat /etc/os-release
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
nvcc --version
python3 --version
python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
nsys --version
ncu --version
```

## 5. 待安装项与路径决策

在继续前需要确认安装路径。本报告不预先锁定具体软件版本；版本应在安装前根据 vLLM、PyTorch 与 NVIDIA 官方兼容矩阵统一选择。

### 方案 A：WSL 系统 CUDA + 独立 Python 虚拟环境（推荐）

- Linux CUDA Toolkit：`/usr/local/cuda-<version>`，并用 `/usr/local/cuda` 软链接选择当前版本
- Python 虚拟环境：`/home/ayax/.venvs/agent-infer-lab`
- Nsight Linux CLI：使用 NVIDIA 的 WSL/Ubuntu 软件包安装到包管理器默认位置
- 项目源码继续保留在 Windows：`/mnt/d/agent-infer-lab`

优点是 CUDA 工具链位置标准、文档和构建脚本兼容性最好，Python 依赖不会污染系统 Python。缺点是需要 `sudo`，且 `/mnt/d` 上进行大量小文件编译通常慢于 WSL ext4。

### 方案 B：Conda/Miniconda 集中管理

- Conda 根目录：`/home/ayax/miniconda3`
- 项目环境：`/home/ayax/miniconda3/envs/agent-infer-lab`
- Nsight 仍建议使用 NVIDIA 系统软件包

优点是 Python、PyTorch 和部分 CUDA 开发包隔离清晰；缺点是 CUDA 编译工具链来源更复杂，排查 vLLM/CUDA Extension 链接问题时变量更多。

## 6. 下一次验收条件

安装完成后必须同时满足：

```text
WSL2 Ubuntu 可启动
/usr/lib/wsl/lib/nvidia-smi 能识别 RTX 4060
nvcc --version 成功
python -c "import torch; print(torch.cuda.is_available())" 输出 True
python -c "import torch; print(torch.version.cuda)" 输出已选 CUDA 版本
nsys --version 成功
ncu --version 成功
```

所有命令的输出与最终安装路径应回填本报告，不能用 Windows 侧工具的存在替代 WSL 侧验收。
