#!/usr/bin/env python3
"""
Inspect one MMFlood sample: shape, channel order, per-channel value ranges,
and a rendered figure (SAR false-colour composite, DEM, and flood mask).

MMFlood is staged manually and never downloaded by project code — see
CLAUDE.md "Data handling". This script assumes data/mmflood/ already
exists and fails loudly, naming the expected path, if it doesn't.

MMFlood is a *geo* dataset: it's indexed by real-world coordinates, not
list position, so `ds[0]` does not work (the original scaffold notes
suggested it — that was wrong). A sampler draws one valid patch instead.

Run:
    uv run python scripts/inspect_mmflood.py
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_ROOT = Path("data/mmflood")
CHANNEL_NAMES = ["VV", "VH", "DEM"]  # order fixed by CLAUDE.md: SAR, then terrain
IGNORE_INDEX = 255  # MMFlood's sentinel for missing-data pixels; excluded from stats
PATCH_SIZE = 512  # pixels; arbitrary for inspection, not a modeling choice


def main() -> None:
    if not DATA_ROOT.exists():
        sys.exit(
            f"No data at {DATA_ROOT}/. MMFlood must be staged manually first "
            "— see README.md 'Setup'. This script never downloads data."
        )

    from torchgeo.datasets import MMFlood
    from torchgeo.samplers import RandomGeoSampler

    print(f"Loading MMFlood from {DATA_ROOT} (train split, DEM included)...")
    ds = MMFlood(root=str(DATA_ROOT), split="train", include_dem=True, download=False)
    print(f"{len(ds)} tiles indexed in the train split\n")

    # MMFlood is geo-indexed (real-world coordinates), not list-indexed —
    # a sampler draws one valid PATCH_SIZE x PATCH_SIZE query instead of ds[0].
    sampler = RandomGeoSampler(ds, size=PATCH_SIZE, length=1)
    query = next(iter(sampler))
    sample = ds[query]

    image, mask = sample["image"], sample["mask"]
    print(f"image shape: {tuple(image.shape)}  (channels, height, width)")
    print(f"mask shape:  {tuple(mask.shape)}\n")

    if image.shape[0] != len(CHANNEL_NAMES):
        sys.exit(
            f"Expected {len(CHANNEL_NAMES)} channels {CHANNEL_NAMES}, "
            f"got {image.shape[0]}. include_dem may not be concatenating "
            "as documented — stop and investigate before trusting anything else."
        )

    for i, name in enumerate(CHANNEL_NAMES):
        band = image[i]
        print(
            f"  {name:<3} min={band.min():.4f}  max={band.max():.4f}  "
            f"mean={band.mean():.4f}  std={band.std():.4f}"
        )

    ignored_fraction = (mask == IGNORE_INDEX).float().mean()
    valid = mask[mask != IGNORE_INDEX]
    flood_fraction = valid.float().mean() if valid.numel() else float("nan")
    print(f"\nignored (missing-data) pixels: {ignored_fraction:.4%}")
    print(f"flood fraction, of valid pixels: {flood_fraction:.4%}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mmflood_sample.png"
    # ds.plot() ships with torchgeo, tailored to MMFlood specifically —
    # reused rather than hand-rolling a figure.
    fig = ds.plot(sample, suptitle="MMFlood sample: VV/VH false-colour, DEM, mask")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
