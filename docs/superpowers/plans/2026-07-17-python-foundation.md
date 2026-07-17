# Python 工程基线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可安装的 Python 包、CPU 可运行的环境检测、pytest/Ruff 质量门禁、GitHub CPU CI 和可复现的 `uv.lock`。

**Architecture:** 正式代码采用 `src/` 布局。环境检测仅使用 Python 标准库，缺少 GPU 时返回结构化“不可用”结果；`--require-gpu` 才将缺少 GPU 视为失败。CPU CI 只同步项目与开发依赖，不安装现有 CUDA/vLLM 环境。

**Tech Stack:** Python 3.12、uv 0.11.29、pytest 9.1.1、Ruff 0.15.22、GitHub Actions。

## Global Constraints

- 新增工具与开发环境必须位于 Ubuntu 的 D 盘 WSL VHD。
- 现有 `/home/ayax/.venvs/agent-infer-lab` GPU 环境不得被同步、清理或改写。
- CI 必须在没有 NVIDIA GPU、CUDA、PyTorch 和 vLLM 的条件下通过。
- `pytest -q` 与 `ruff check .` 必须全部通过。
- 只锁定当前代码实际使用的项目与开发依赖；GPU 依赖在代码首次使用时作为独立可选组加入。

---

### Task 1: Python 包与测试基线

**Files:**
- Create: `pyproject.toml`
- Create: `src/agent_infer_lab/__init__.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Produces: 可通过 `import agent_infer_lab` 导入的包，公开 `__version__ == "0.1.0"`。

- [ ] **Step 1: 写失败测试**

```python
def test_package_exposes_version() -> None:
    import agent_infer_lab

    assert agent_infer_lab.__version__ == "0.1.0"
```

- [ ] **Step 2: 验证测试因包不存在而失败**

Run: `/home/ayax/.venvs/agent-infer-lab-dev/bin/python -m pytest -q`

Expected: `ModuleNotFoundError: No module named 'agent_infer_lab'`。

- [ ] **Step 3: 添加最小包与项目配置**

`pyproject.toml` 声明 Python 3.12、hatchling 构建后端、pytest/Ruff 开发依赖和工具配置；`__init__.py` 只定义版本。

- [ ] **Step 4: 同步开发环境并生成锁文件**

Run: `UV_PROJECT_ENVIRONMENT=/home/ayax/.venvs/agent-infer-lab-dev UV_CACHE_DIR=/home/ayax/.cache/uv /home/ayax/.local/bin/uv sync`

Expected: 生成 `uv.lock`，项目以 editable 方式安装。

- [ ] **Step 5: 验证包测试通过**

Run: `UV_PROJECT_ENVIRONMENT=/home/ayax/.venvs/agent-infer-lab-dev /home/ayax/.local/bin/uv run --locked pytest -q`

Expected: `1 passed`。

### Task 2: CPU 安全的环境检测

**Files:**
- Create: `src/agent_infer_lab/environment.py`
- Create: `scripts/check_environment.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Produces: `collect_environment()` 返回字典；`main(argv)` 支持 `--json` 与 `--require-gpu`。

- [ ] **Step 1: 为无 GPU 行为写失败测试**

测试要求默认模式在 `gpu.available == False` 时返回 0，严格模式返回 1；JSON 输出必须包含 Python、平台、包版本和工具状态。

- [ ] **Step 2: 验证测试因模块不存在而失败**

Run: `/home/ayax/.venvs/agent-infer-lab-dev/bin/python -m pytest tests/test_environment.py -q`

Expected: 导入 `agent_infer_lab.environment` 失败。

- [ ] **Step 3: 实现最小标准库检测器**

检测 Python/平台、`torch`/`vllm` 包版本、`nvcc`/`nsys`/`ncu` 命令和 `nvidia-smi`。所有外部命令设置超时，异常转为结构化不可用状态。

- [ ] **Step 4: 验证 CPU 行为通过**

Run: `/home/ayax/.venvs/agent-infer-lab-dev/bin/python -m pytest tests/test_environment.py -q`

Expected: 所有环境检测测试通过。

### Task 3: Ruff、CPU CI 与完整验收

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/progress/2026-07-17.md`

**Interfaces:**
- Consumes: `pyproject.toml`、`uv.lock`、全部测试和环境检测 CLI。
- Produces: push/PR 时运行的 CPU-only 质量门禁。

- [ ] **Step 1: 添加 CPU CI**

CI 使用 Ubuntu 与 Python 3.12，执行 `uv sync --locked`、`uv run --locked pytest -q`、`uv run --locked ruff check .` 和 CPU 环境检测。工作流不得安装或调用 GPU extra。

- [ ] **Step 2: 验证锁文件未漂移**

Run: `/home/ayax/.local/bin/uv lock --check`

Expected: exit 0。

- [ ] **Step 3: 运行完整测试和静态检查**

Run: `uv run --locked pytest -q`

Expected: 全部测试通过。

Run: `uv run --locked ruff check .`

Expected: `All checks passed!`。

- [ ] **Step 4: 模拟 CPU CI 环境检测**

Run: `uv run --locked python scripts/check_environment.py --json`

Expected: exit 0；缺少 GPU/PyTorch/vLLM 时仅报告不可用。

- [ ] **Step 5: 更新当日日志并提交**

显式暂存本计划涉及文件，提交信息使用 `build: establish Python quality baseline`。
