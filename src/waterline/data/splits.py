"""Split-selection logic for MMFlood.

Deliberately separated from any file/raster I/O (see mmflood.py for that),
so the actual split-group logic can be unit-tested without needing the
real, un-vendored dataset present. See tests/test_splits.py.

Two strategies, both documented as a hard project rule in CLAUDE.md:

- "event": MMFlood's own train/val/test assignment. Verified directly
  against the real dataset metadata (95 flood events, one subset each,
  zero events appearing in more than one subset) -- this is already
  event-disjoint, so no extra grouping logic is needed for it.
- "random": deliberately ignores event boundaries. Pools every tile from
  the train+test events together and reassigns them randomly, at the
  individual-tile level -- reproducing the exact leakage a naive random
  split would cause, so the resulting score gap is real evidence, not an
  assumption.
"""

from __future__ import annotations

import random
from typing import Literal

SplitStrategy = Literal["event", "random"]


def random_tile_split(
    tile_ids: list[str], n_train: int, seed: int
) -> tuple[list[str], list[str]]:
    """Partition tile_ids into (train, test) groups of size (n_train, rest).

    Ignores any structure in tile_ids (e.g. which event a tile came from)
    by design -- this is what makes it the "leaky" comparison split, not a
    bug to fix. Sizes are chosen by the caller to match the event-wise
    split's tile counts, so the only thing that differs between the two
    experiments is the split rule itself, not how much data each side gets.

    Args:
        tile_ids: every tile identifier in the pool (train+test events
            combined; val is kept out of this pool entirely).
        n_train: how many tiles go to the train group; the rest go to test.
        seed: makes the partition reproducible across runs.

    Raises:
        ValueError: if n_train is not a valid size for tile_ids.
    """
    if not 0 <= n_train <= len(tile_ids):
        raise ValueError(f"n_train={n_train} out of range for {len(tile_ids)} tiles")
    pool = list(tile_ids)
    random.Random(seed).shuffle(pool)
    return pool[:n_train], pool[n_train:]


def event_of(tile_id: str) -> str:
    """The EMS activation (event) ID a tile identifier belongs to.

    Tile identifiers look like "EMSR446-0-1" (event-AOI-tileindex); the
    event is everything before the first "-".
    """
    return tile_id.split("-")[0]
