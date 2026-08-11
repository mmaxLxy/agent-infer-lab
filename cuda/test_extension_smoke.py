"""Compile and execute the first AgentInferLab CUDA kernel."""

import torch
from load_extension import load_kv_cache_extension


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this smoke test")

    extension = load_kv_cache_extension()

    slots = torch.tensor(
        [0, 15, 16, 17, 35, 63],
        dtype=torch.int64,
        device="cuda",
    )
    expected_locations = torch.tensor(
        [
            [0, 0],
            [0, 15],
            [1, 0],
            [1, 1],
            [2, 3],
            [3, 15],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    actual_locations = extension.resolve_slots(
        slots,
        16,
    )

    torch.testing.assert_close(
        actual_locations,
        expected_locations,
        rtol=0,
        atol=0,
    )

    empty_slots = torch.empty(
        0,
        dtype=torch.int64,
        device="cuda",
    )
    empty_locations = extension.resolve_slots(
        empty_slots,
        16,
    )

    assert tuple(empty_locations.shape) == (0, 2)

    print("CUDA extension smoke test passed")
    print("slots:", slots.cpu().tolist())
    print(
        "locations:",
        actual_locations.cpu().tolist(),
    )


if __name__ == "__main__":
    main()
