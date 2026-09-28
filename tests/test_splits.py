"""Split-selection logic tests. No real MMFlood data needed -- these test
splits.py's pure functions directly, in isolation from any file I/O.
"""

import pytest

from waterline.data.splits import event_of, random_tile_split


def test_event_of_extracts_event_prefix() -> None:
    assert event_of("EMSR446-0-1") == "EMSR446"
    assert event_of("EMSR107-12-3") == "EMSR107"


def test_random_tile_split_sizes() -> None:
    tile_ids = [f"EVT{i}-0-0" for i in range(100)]
    train, test = random_tile_split(tile_ids, n_train=70, seed=0)
    assert len(train) == 70
    assert len(test) == 30


def test_random_tile_split_disjoint_and_complete() -> None:
    """The core correctness property: no tile is lost or duplicated."""
    tile_ids = [f"EVT{i}-0-0" for i in range(50)]
    train, test = random_tile_split(tile_ids, n_train=30, seed=1)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(tile_ids)


def test_random_tile_split_is_reproducible() -> None:
    tile_ids = [f"EVT{i}-0-0" for i in range(50)]
    train_a, test_a = random_tile_split(tile_ids, n_train=30, seed=42)
    train_b, test_b = random_tile_split(tile_ids, n_train=30, seed=42)
    assert train_a == train_b
    assert test_a == test_b


def test_random_tile_split_deliberately_mixes_events() -> None:
    """The whole point of this split strategy: unlike an event-wise split,
    it does NOT keep a single event's tiles together. This test proves the
    two strategies actually differ, rather than assuming it.
    """
    # 10 events, 10 tiles each -- large enough that an even split landing
    # every event entirely on one side by chance is effectively impossible.
    tile_ids = [f"EVT{e}-0-{t}" for e in range(10) for t in range(10)]
    train, test = random_tile_split(tile_ids, n_train=50, seed=0)

    train_events = {event_of(t) for t in train}
    test_events = {event_of(t) for t in test}
    assert train_events & test_events, (
        "expected at least one event to have tiles on both sides -- "
        "if this fails, the random split isn't actually mixing events"
    )


def test_random_tile_split_rejects_invalid_n_train() -> None:
    tile_ids = ["a", "b", "c"]
    with pytest.raises(ValueError):
        random_tile_split(tile_ids, n_train=10, seed=0)
    with pytest.raises(ValueError):
        random_tile_split(tile_ids, n_train=-1, seed=0)
