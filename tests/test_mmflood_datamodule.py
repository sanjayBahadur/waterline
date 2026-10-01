"""Tests against the real, un-vendored MMFlood dataset.

Marked slow and skipped by default (see pyproject.toml) -- CI has no
access to the real 47GB dataset. Run locally with:

    uv run pytest -m slow
"""

from pathlib import Path

import pytest
import torch
from torchgeo.samplers import GridGeoSampler, RandomPatchSampler

from waterline.data.mmflood import (
    CHANNEL_NAMES,
    IGNORE_INDEX,
    WaterlineDataModule,
    event_ids_for_subset,
    split_for_events,
)
from waterline.data.splits import SplitStrategy

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "mmflood"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="MMFlood not staged locally (see CLAUDE.md)"
)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["event", "random"])
def test_datamodule_train_batch_shape(strategy: SplitStrategy) -> None:
    dm = WaterlineDataModule(
        data_root=DATA_ROOT, split_strategy=strategy, patch_size=128, batch_size=4
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (4, len(CHANNEL_NAMES), 128, 128)
    assert batch["mask"].shape == (4, 128, 128)


@pytest.mark.slow
def test_datamodule_split_sizes_match_between_strategies() -> None:
    """Total tile counts must be identical between strategies -- only the
    split *rule* should differ, per CLAUDE.md's ablation principle, not
    how much data either side gets.
    """
    sizes = {}
    strategies: list[SplitStrategy] = ["event", "random"]
    for strategy in strategies:
        dm = WaterlineDataModule(data_root=DATA_ROOT, split_strategy=strategy)
        dm.setup()
        sizes[strategy] = (
            len(dm.train_dataset),
            len(dm.val_dataset),
            len(dm.test_dataset),
        )
    assert sizes["event"] == sizes["random"]


@pytest.mark.slow
def test_no_nan_pixels_survive_across_full_test_split() -> None:
    """Some source tiles have genuine sensor gaps -- caught only by
    scanning the *entire* test split with Otsu (#7); a handful of
    spot-checked patches earlier happened not to hit one. MMFloodSplit's
    __getitem__ must zero those image pixels and mark their mask
    ignore_index, exactly like MMFlood's own __getitem__ does -- not
    something inherited for free from IntersectionDataset.
    """
    events = event_ids_for_subset(DATA_ROOT, "test")
    dataset = split_for_events(DATA_ROOT, events, include_dem=True)
    sampler = GridGeoSampler(dataset, size=256, stride=256)

    for query in sampler:
        sample = dataset[query]
        assert not torch.isnan(sample["image"]).any(), (
            "NaN pixel survived into a sample -- missing-data cleanup broke"
        )
        mask_values = torch.unique(sample["mask"])
        assert set(mask_values.tolist()) <= {0, 1, IGNORE_INDEX}


@pytest.mark.slow
def test_augmentation_preserves_mask_labels() -> None:
    """Flips/rotations must use nearest-neighbour resampling for the mask,
    not bilinear -- otherwise 0/1 labels blur into meaningless fractional
    values at boundaries. This is the exact bug caught while building this
    module; kept as a regression test.
    """
    dm = WaterlineDataModule(
        data_root=DATA_ROOT, split_strategy="event", patch_size=128, batch_size=4
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    batch["image"] = dm._normalize(batch["image"])

    augmented = dm.train_aug(dict(batch))
    values = torch.unique(augmented["mask"])
    assert set(values.tolist()) <= {0, 1, 255}, (
        f"expected only label values {{0, 1, 255}}, got {values.tolist()} -- "
        "augmentation is likely interpolating the mask instead of using "
        "nearest-neighbour resampling"
    )


@pytest.mark.slow
@pytest.mark.parametrize("include_dem", [False, True])
def test_sampler_draws_survive_a_full_epoch(include_dem: bool) -> None:
    """Regression test for a real bug caught during actual Colab training:
    MMFloodSplit used to override self.index (set correctly by
    IntersectionDataset's own gpd.overlay-based intersection) with just
    self.image.index -- a cruder approximation that doesn't account for
    tiny floating-point boundary differences between a tile's paired
    image/DEM/mask files. A sampler built from that approximate index
    could draw a point valid for the image but not exactly matching the
    mask's true footprint, raising IndexError deep in torchgeo's own
    IntersectionDataset.__getitem__ and crashing training outright.

    Draws a large, bounded sample of random patches -- the failure was rare
    enough that a handful of draws didn't reliably catch it either before
    or after the fix. (Not the full epoch size: that's already been
    verified manually at 1200 draws/config with 0 failures; this stays
    smaller so the slow suite finishes in reasonable time.)
    """
    events = event_ids_for_subset(DATA_ROOT, "train")
    dataset = split_for_events(DATA_ROOT, events, include_dem)
    n_draws = min(400, len(dataset))
    sampler = RandomPatchSampler(dataset, size=256, length=n_draws)

    failures = 0
    for query in sampler:
        try:
            dataset[query]
        except IndexError:
            failures += 1
    assert failures == 0, f"{failures}/{n_draws} sampler draws raised IndexError"
