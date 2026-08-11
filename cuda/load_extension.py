"""Build and load the local paged KV cache CUDA extension."""

from pathlib import Path

from torch.utils.cpp_extension import load

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CUDA_DIRECTORY = _PROJECT_ROOT / "cuda"


def load_kv_cache_extension(*, verbose: bool = True):
    """Compile and load the C++/CUDA extension."""

    return load(
        name="agent_infer_lab_kv_cache_cuda",
        sources=[
            str(_CUDA_DIRECTORY / "kv_cache_ops.cpp"),
            str(_CUDA_DIRECTORY / "kv_cache_kernels.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
        verbose=verbose,
    )
