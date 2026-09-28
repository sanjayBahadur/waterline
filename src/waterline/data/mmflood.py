"""Lightning DataModule over MMFlood, with a choice of split strategy.

Two ways to build the train/test split, switched by `split_strategy`:

  "event"  (default, the honest number) -- MMFlood's own train/val/test
           assignment. Verified against the real dataset metadata: 95
           events, one subset each, none split across subsets -- so this
           needs no extra grouping logic, just using MMFlood's own split.

  "random" (the comparison number, expected to be inflated) -- pools every
           tile from the train+test events together and reassigns them at
           the individual-tile level, ignoring which event each tile came
           from. Same total tile counts as "event" on both sides, so the
           split rule is the only thing that differs -- see
           CLAUDE.md's hard rule #1 and splits.py's docstring.

Channel order: VV, VH, then DEM if include_dem=True (CLAUDE.md rule #2).
Normalization: torchgeo's own precomputed MMFlood median/std, the same
ones already used in scripts/smoke_overfit.py -- reused, not retyped.
Augmentation: horizontal/vertical flip and 90-degree rotation only, train
split only. No colour jitter -- meaningless on backscatter (CLAUDE.md).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import kornia.augmentation as K  # noqa: N812 -- standard alias, matches torchgeo's own usage
import lightning as pl
import pandas as pd
import torch
from kornia.constants import DataKey, Resample
from torch.utils.data import DataLoader
from torchgeo.datamodules.mmflood import MMFloodDataModule as _TorchgeoMMFloodStats
from torchgeo.datasets.geo import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler, RandomPatchSampler

from waterline.data.splits import SplitStrategy, random_tile_split

METADATA_FILE = "activations.json"
CHANNEL_NAMES = ["VV", "VH", "DEM"]


def _event_ids(root: Path, subset: str) -> list[str]:
    """Event IDs MMFlood itself assigns to a subset (train/val/test)."""
    df = pd.read_json(root / METADATA_FILE).transpose()
    # pandas' .tolist() is untyped (returns Any) -- cast for mypy strict.
    return [str(event_id) for event_id in df[df["subset"] == subset].index]


def _tile_paths(root: Path, event_ids: list[str], content: str) -> list[str]:
    """All .tif paths for one content type, for the given events.

    Mirrors torchgeo's own MMFloodComponent glob pattern exactly
    (root/**/{event}*-*/content/*.tif) -- verified against the real
    dataset directory structure before writing this, not assumed.
    """
    paths: list[str] = []
    for event in event_ids:
        paths += glob(str(root / "**" / f"{event}*-*" / content / "*.tif"))
    return sorted(paths)


class _ExplicitPathsComponent(RasterDataset):
    """Like torchgeo's MMFloodComponent, but from an explicit file list
    instead of globbing by event-ID prefix.

    Needed for the random-tile split: MMFloodComponent's glob pattern only
    understands whole-event prefixes (root/**/{event}*-*/...), so pooling
    and reshuffling individual tiles across events -- which is the entire
    point of the random-split comparison -- can't be expressed through it.
    RasterDataset itself accepts an explicit path list directly (verified
    against its source), so this just supplies that instead of a glob.
    """

    def __init__(self, paths: list[str], content: str) -> None:
        self.content = content
        self.is_image = content != "mask"
        super().__init__(paths)


class MMFloodSplit(IntersectionDataset):
    """One train/val/test split of MMFlood, built from an explicit tile
    selection -- the same construction MMFlood.__init__ does internally,
    generalised to accept any tile selection rather than one tied to whole
    events. See module docstring for why this is needed.
    """

    def __init__(
        self,
        s1_paths: list[str],
        mask_paths: list[str],
        dem_paths: list[str] | None = None,
    ) -> None:
        image: RasterDataset | IntersectionDataset = _ExplicitPathsComponent(
            s1_paths, "s1_raw"
        )
        if dem_paths is not None:
            dem = _ExplicitPathsComponent(dem_paths, "DEM")
            image = image & dem
            image.index = dem.index
        self.image = image
        self.mask = _ExplicitPathsComponent(mask_paths, "mask")
        super().__init__(self.image, self.mask)
        self.index = self.image.index


def _split_for_events(
    root: Path, event_ids: list[str], include_dem: bool
) -> MMFloodSplit:
    s1_paths = _tile_paths(root, event_ids, "s1_raw")
    dem_paths = _tile_paths(root, event_ids, "DEM") if include_dem else None
    mask_paths = _tile_paths(root, event_ids, "mask")
    return MMFloodSplit(s1_paths, mask_paths, dem_paths)


def _split_for_tile_ids(
    root: Path, event_ids: list[str], tile_ids: set[str], include_dem: bool
) -> MMFloodSplit:
    """Like _split_for_events, but filtered down to a specific tile-ID
    subset -- used by the random split, which pools tiles from many events
    and only wants some of each event's tiles in a given group.
    """

    def _filter(paths: list[str]) -> list[str]:
        return [p for p in paths if Path(p).stem in tile_ids]

    s1_paths = _filter(_tile_paths(root, event_ids, "s1_raw"))
    dem_paths = _filter(_tile_paths(root, event_ids, "DEM")) if include_dem else None
    mask_paths = _filter(_tile_paths(root, event_ids, "mask"))
    return MMFloodSplit(s1_paths, mask_paths, dem_paths)


class WaterlineDataModule(pl.LightningDataModule):
    """MMFlood, with a choice of event-wise (honest) or random (leaky
    comparison) train/test split. See module docstring.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_strategy: SplitStrategy = "event",
        include_dem: bool = True,
        patch_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 0,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.split_strategy = split_strategy
        self.include_dem = include_dem
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # RandomPatchSampler's own length=None default estimates the max
        # number of non-overlapping patches from raw geometric area, which
        # for this dataset's many separate, real-world-sized rasters comes
        # out to tens of thousands -- and it materializes that many sample
        # points eagerly before training even starts (confirmed: ~60k
        # points made iter() hang for 100+ seconds; explicit 1242 took
        # ~3s). Default instead to one draw per source tile, a normal
        # "roughly one look per training epoch" convention.
        self.samples_per_epoch = samples_per_epoch

        n_channels = len(CHANNEL_NAMES) if include_dem else 2
        self.norm_median = _TorchgeoMMFloodStats.median[:n_channels].view(1, -1, 1, 1)
        self.norm_std = _TorchgeoMMFloodStats.std[:n_channels].view(1, -1, 1, 1)

        # data_keys=None: kornia infers "image" vs "mask" from the batch
        # dict itself, and -- this is the part that matters -- applies the
        # *same* sampled flip/rotation to both, so they stay spatially
        # aligned. extra_args forces nearest-neighbour resampling for the
        # mask specifically (bilinear would blur 0/1/255 labels into
        # meaningless fractional values). Applied to the whole batch dict
        # in on_after_batch_transfer below, never to the image tensor alone.
        self.train_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation90((0, 3), p=0.5),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

        self.train_dataset: MMFloodSplit
        self.val_dataset: MMFloodSplit
        self.test_dataset: MMFloodSplit

    def setup(self, stage: str | None = None) -> None:
        val_events = _event_ids(self.root, "val")
        self.val_dataset = _split_for_events(self.root, val_events, self.include_dem)

        if self.split_strategy == "event":
            train_events = _event_ids(self.root, "train")
            test_events = _event_ids(self.root, "test")
            self.train_dataset = _split_for_events(
                self.root, train_events, self.include_dem
            )
            self.test_dataset = _split_for_events(
                self.root, test_events, self.include_dem
            )
            return

        # "random": pool train+test tiles, reshuffle at the tile level,
        # keeping the same group sizes as the event-wise split so total
        # data volume isn't a confound -- only the split rule changes.
        train_events = _event_ids(self.root, "train")
        test_events = _event_ids(self.root, "test")
        pool_events = train_events + test_events
        pool_tile_ids = [
            Path(p).stem for p in _tile_paths(self.root, pool_events, "s1_raw")
        ]
        n_train = len(_tile_paths(self.root, train_events, "s1_raw"))
        train_ids, test_ids = random_tile_split(pool_tile_ids, n_train, self.seed)
        self.train_dataset = _split_for_tile_ids(
            self.root, pool_events, set(train_ids), self.include_dem
        )
        self.test_dataset = _split_for_tile_ids(
            self.root, pool_events, set(test_ids), self.include_dem
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.norm_median.to(image.device)) / self.norm_std.to(
            image.device
        )

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        batch["image"] = self._normalize(batch["image"])
        if self.trainer is not None and self.trainer.training:
            # The whole dict, not batch["image"] alone -- kornia samples one
            # random flip/rotation and applies it to both "image" and
            # "mask" together, keeping them aligned. Augmenting only the
            # image would silently shift flood labels off the water they
            # label.
            batch = self.train_aug(batch)
        return batch

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        length = self.samples_per_epoch or len(self.train_dataset)
        sampler = RandomPatchSampler(
            self.train_dataset, size=self.patch_size, length=length
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
        )

    def _eval_dataloader(
        self, dataset: MMFloodSplit
    ) -> DataLoader[dict[str, torch.Tensor]]:
        # Grid, not random: eval must be deterministic and cover every
        # pixel exactly once, not an arbitrary random subset.
        sampler = GridGeoSampler(dataset, size=self.patch_size, stride=self.patch_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._eval_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._eval_dataloader(self.test_dataset)
