#!/usr/bin/env python3
"""
M0 exit criterion (issue #3): prove the training pipeline is wired
correctly by overfitting a U-Net to a single batch until the loss
collapses toward zero, and record a seconds/epoch estimate on the full
train split.

This is deliberately throwaway. It says nothing about generalisation --
only that data -> model -> loss -> gradient update are actually
connected. It is NOT the real M1 datamodule or baseline model; those
get built properly in their own issues, with event-wise splits and a
considered loss/architecture choice.

Decisions already made for this smoke test (see .userlogs.txt):
  - Channels: VV, VH, DEM (3) -- deliberately includes the terrain
    channel early, ahead of the M2 ablation, per project decision.
  - Normalization: torchgeo's own precomputed MMFlood median/std,
    reused directly from the library rather than retyped by hand, so
    it can't silently drift out of sync.
  - Loss: plain BCEWithLogitsLoss, a placeholder. The real loss
    function (with class weighting) is decided separately at the M1
    baseline issue.
  - Encoder: resnet18, weights=None (random init). No pretrained
    download needed to keep this test hermetic and fast; a pretrained
    encoder is worth considering for the real M1 baseline, not decided
    here.

Run:
    uv run python scripts/smoke_overfit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "mmflood"
CHANNEL_NAMES = ["VV", "VH", "DEM"]
IGNORE_INDEX = 255  # MMFlood's sentinel for missing-data pixels; excluded from loss

PATCH_SIZE = 256  # pixels; smaller than the inspection script's 512, for speed
BATCH_SIZE = 4
MAX_EPOCHS = 150  # "epoch" here means one more pass over the *same* single batch
PROBE_BATCHES = 5  # fresh (non-overfit) batches used to estimate seconds/epoch


def main() -> None:
    if not DATA_ROOT.exists():
        sys.exit(
            f"No data at {DATA_ROOT}/. MMFlood must be staged manually first "
            "— see README.md 'Setup'. This script never downloads data."
        )

    import lightning as pl
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader
    from torchgeo.datamodules.mmflood import MMFloodDataModule
    from torchgeo.datasets import MMFlood
    from torchgeo.datasets.utils import stack_samples
    from torchgeo.samplers import RandomPatchSampler

    # First 3 of torchgeo's 4 precomputed stats (VV, VH, DEM); drops the 4th
    # (hydro), which this smoke test doesn't load.
    norm_median = MMFloodDataModule.median[: len(CHANNEL_NAMES)].view(1, -1, 1, 1)
    norm_std = MMFloodDataModule.std[: len(CHANNEL_NAMES)].view(1, -1, 1, 1)

    class SmokeUNet(pl.LightningModule):
        """Throwaway U-Net for the overfit check. Not the real M1 baseline."""

        def __init__(self) -> None:
            super().__init__()
            self.model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=len(CHANNEL_NAMES),
                classes=1,
            )
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.loss_history: list[float] = []

        def _step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            image = (batch["image"] - norm_median) / norm_std
            mask = batch["mask"]
            valid = mask != IGNORE_INDEX
            target = (mask == 1).float()
            logits = self.model(image).squeeze(1)
            return self.loss_fn(logits[valid], target[valid])

        def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
            # Called from a plain manual loop below, not trainer.fit() (see
            # why), so self.log() isn't available here -- no Trainer is
            # attached. loss_history is our own tracking instead.
            loss = self._step(batch)
            self.loss_history.append(loss.item())
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    print(f"Loading MMFlood from {DATA_ROOT} (train split, DEM included)...")
    ds = MMFlood(root=str(DATA_ROOT), split="train", include_dem=True, download=False)
    print(f"{len(ds)} tiles indexed in the train split\n")

    sampler = RandomPatchSampler(ds, size=PATCH_SIZE, length=BATCH_SIZE)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=stack_samples
    )

    # --- Part 1: overfit a single batch ------------------------------------
    # Lightning's Trainer(overfit_batches=1) is the textbook way to do this,
    # but it doesn't work here: it re-indexes the dataloader using plain
    # integer positions, and torchgeo's geo-samplers don't support that (an
    # MMFlood tile isn't "item 7," it's a real-world coordinate query) --
    # confirmed by actually running it, which crashed inside torchgeo's
    # indexing code. Worked around with a manual loop over one frozen batch
    # instead. The real M1 datamodule should use torchgeo's own
    # GeoDataModule/MMFloodDataModule machinery, which is built to
    # interoperate with the Trainer correctly -- worth remembering when
    # that issue comes up.
    print(f"Overfitting one batch of {BATCH_SIZE} for {MAX_EPOCHS} steps...")
    model = SmokeUNet()
    optimizer = model.configure_optimizers()
    fixed_batch = next(iter(train_loader))
    model.train()
    for _ in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = model.training_step(fixed_batch, 0)
        loss.backward()
        optimizer.step()

    first_loss = model.loss_history[0]
    last_loss = model.loss_history[-1]
    print(
        f"\nloss: {first_loss:.4f} -> {last_loss:.4f} over {len(model.loss_history)} steps"
    )
    if last_loss >= first_loss * 0.1:
        print(
            "WARNING: loss did not collapse toward zero. The pipeline may be "
            "wired correctly but not actually memorizing -- investigate before "
            "trusting this as a pass."
        )
    else:
        print("Loss collapsed as expected -- pipeline is wired correctly.")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.loss_history)
    ax.set_xlabel("step")
    ax.set_ylabel("BCE loss")
    ax.set_title("Single-batch overfit — smoke test")
    fig.savefig(out_dir / "smoke_overfit_loss.png", dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {out_dir / 'smoke_overfit_loss.png'}")

    # --- Part 2: seconds/epoch estimate on the full train split ------------
    print(f"\nTiming {PROBE_BATCHES} fresh batches to estimate full-epoch duration...")
    probe_sampler = RandomPatchSampler(
        ds, size=PATCH_SIZE, length=PROBE_BATCHES * BATCH_SIZE
    )
    probe_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=probe_sampler, collate_fn=stack_samples
    )
    probe_model = SmokeUNet()
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
    probe_model.train()

    batch_times: list[float] = []
    for batch in probe_loader:
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = probe_model._step(batch)
        loss.backward()
        optimizer.step()
        batch_times.append(time.perf_counter() - start)

    avg_batch_time = sum(batch_times) / len(batch_times)
    batches_per_epoch = -(-len(ds) // BATCH_SIZE)  # ceiling division
    estimated_epoch_seconds = avg_batch_time * batches_per_epoch
    print(f"avg seconds/batch (batch_size={BATCH_SIZE}): {avg_batch_time:.3f}")
    print(f"batches/epoch over {len(ds)} tiles: {batches_per_epoch}")
    print(f"estimated seconds/epoch: {estimated_epoch_seconds:.1f}")


if __name__ == "__main__":
    main()
