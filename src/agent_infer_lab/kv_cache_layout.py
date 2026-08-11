"""Pure CPU definitions for paged KV cache layouts and slot mapping."""

from dataclasses import dataclass


def _require_positive_integer(name: str, value: object) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _require_valid_slot(slot: object, capacity_tokens: int) -> None:
    if not isinstance(slot, int) or isinstance(slot, bool):
        raise ValueError("slot must be an integer")
    if slot < 0 or slot >= capacity_tokens:
        raise ValueError(
            f"slot must be in [0, {capacity_tokens}), got {slot}"
        )


@dataclass(frozen=True)
class SlotLocation:
    """Physical block location corresponding to one linear cache slot."""

    slot: int
    block_id: int
    block_offset: int


@dataclass(frozen=True)
class PagedKVCacheLayout:
    """Shape and indexing rules for one layer of paged KV cache."""

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("block_size", self.block_size),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            _require_positive_integer(name, value)

    @property
    def capacity_tokens(self) -> int:
        """Return the total number of token slots in the cache."""

        return self.num_blocks * self.block_size

    @property
    def values_per_token(self) -> int:
        """Return the number of scalar K or V values stored per token."""

        return self.num_kv_heads * self.head_dim

    def locate(self, slot: int) -> SlotLocation:
        """Convert one linear slot into a physical block and block offset."""

        _require_valid_slot(slot, self.capacity_tokens)
        return SlotLocation(
            slot=slot,
            block_id=slot // self.block_size,
            block_offset=slot % self.block_size,
        )

    def locate_many(
        self,
        slot_mapping: tuple[int, ...],
        *,
        require_unique: bool = False,
    ) -> tuple[SlotLocation, ...]:
        """Resolve multiple slots while preserving their logical order."""

        locations: list[SlotLocation] = []
        seen_slots: set[int] = set()

        for slot in slot_mapping:
            if require_unique and slot in seen_slots:
                raise ValueError(
                    f"slot_mapping contains duplicate slot {slot}"
                )

            location = self.locate(slot)
            locations.append(location)
            seen_slots.add(slot)

        return tuple(locations)
