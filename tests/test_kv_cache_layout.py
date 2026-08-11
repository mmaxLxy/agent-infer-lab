from dataclasses import FrozenInstanceError

import pytest

from agent_infer_lab.kv_cache_layout import (
    PagedKVCacheLayout,
    SlotLocation,
)


def make_layout() -> PagedKVCacheLayout:
    return PagedKVCacheLayout(
        num_blocks=4,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
    )


def test_layout_records_are_immutable() -> None:
    layout = make_layout()
    location = layout.locate(35)

    with pytest.raises(FrozenInstanceError):
        layout.block_size = 32  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        location.block_id = 1  # type: ignore[misc]


def test_layout_calculates_capacity_and_values_per_token() -> None:
    layout = make_layout()

    assert layout.capacity_tokens == 64
    assert layout.values_per_token == 128


@pytest.mark.parametrize(
    ("slot", "expected_block", "expected_offset"),
    [
        (0, 0, 0),
        (15, 0, 15),
        (16, 1, 0),
        (17, 1, 1),
        (35, 2, 3),
        (63, 3, 15),
    ],
)
def test_locate_maps_linear_slots_to_physical_blocks(
    slot: int,
    expected_block: int,
    expected_offset: int,
) -> None:
    location = make_layout().locate(slot)

    assert location == SlotLocation(
        slot=slot,
        block_id=expected_block,
        block_offset=expected_offset,
    )


def test_layout_matches_qwen_kv_shape() -> None:
    layout = PagedKVCacheLayout(
        num_blocks=128,
        block_size=16,
        num_kv_heads=2,
        head_dim=64,
    )

    assert layout.capacity_tokens == 2048
    assert layout.values_per_token == 128
    assert layout.locate(1536) == SlotLocation(
        slot=1536,
        block_id=96,
        block_offset=0,
    )
    assert layout.locate(2047) == SlotLocation(
        slot=2047,
        block_id=127,
        block_offset=15,
    )


def test_locate_many_preserves_input_order() -> None:
    locations = make_layout().locate_many((35, 0, 16))

    assert locations == (
        SlotLocation(slot=35, block_id=2, block_offset=3),
        SlotLocation(slot=0, block_id=0, block_offset=0),
        SlotLocation(slot=16, block_id=1, block_offset=0),
    )


def test_locate_many_allows_duplicate_reads_by_default() -> None:
    locations = make_layout().locate_many((17, 17))

    assert locations == (
        SlotLocation(slot=17, block_id=1, block_offset=1),
        SlotLocation(slot=17, block_id=1, block_offset=1),
    )


def test_locate_many_rejects_duplicate_writes_when_required() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate slot 17",
    ):
        make_layout().locate_many(
            (17, 17),
            require_unique=True,
        )


def test_locate_many_accepts_an_empty_no_op() -> None:
    assert make_layout().locate_many(()) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_blocks", 0),
        ("num_blocks", -1),
        ("block_size", 0),
        ("block_size", True),
        ("num_kv_heads", 0),
        ("num_kv_heads", 2.5),
        ("head_dim", 0),
        ("head_dim", False),
    ],
)
def test_layout_rejects_invalid_dimensions(
    field: str,
    value: object,
) -> None:
    dimensions: dict[str, object] = {
        "num_blocks": 4,
        "block_size": 16,
        "num_kv_heads": 2,
        "head_dim": 64,
    }
    dimensions[field] = value

    with pytest.raises(ValueError):
        PagedKVCacheLayout(**dimensions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "slot",
    [
        -1,
        64,
        True,
        1.5,
    ],
)
def test_locate_rejects_invalid_slots(slot: object) -> None:
    with pytest.raises(ValueError):
        make_layout().locate(slot)  # type: ignore[arg-type]
