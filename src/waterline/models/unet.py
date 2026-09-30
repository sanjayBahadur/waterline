"""U-Net baseline (issue #8).

VV + VH only, deliberately -- not VV/VH/DEM. This is the M1 baseline
that must stay architecturally identical to what M2's terrain ablation
trains, with only the input channels changing (CLAUDE.md's hard rule
#5: "the terrain ablation must be a clean ablation"). Including DEM
here now would break that later.

Loss: Dice + BCE, unweighted. Dice already handles the dataset's ~10:1
class imbalance (it scores region overlap, not raw per-pixel accuracy),
so an added pos_weight on BCE would just push harder toward exactly the
failure mode the Otsu baseline already showed (#7: 97% recall, 3.6%
precision, over-predicting flood everywhere). Decided with the project
owner; see .userlogs.txt.

Encoder: resnet34, ImageNet-pretrained. Meant for real GPU training
(Colab) -- unlike the M0 smoke test's resnet18 with random init, which
was deliberately hermetic and fast for a plumbing check, not a real run.
"""

from __future__ import annotations

from typing import cast

import lightning as pl
import segmentation_models_pytorch as smp
import torch
import wandb

from waterline.data.mmflood import IGNORE_INDEX
from waterline.eval.metrics import SegmentationMetrics

IN_CHANNELS = 2  # VV, VH -- DEM is M2's job, not this baseline's


class FloodUNet(pl.LightningModule):
    """Lightning wrapper around segmentation_models_pytorch's U-Net.

    training_step/validation_step/test_step all use the same masked
    Dice+BCE loss. Validation and test additionally accumulate
    SegmentationMetrics (IoU/F1/precision/recall -- the same accumulator
    the evaluation harness uses, see eval/metrics.py) and log them at
    each epoch's end.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=IN_CHANNELS,
            classes=1,
        )
        self.dice_loss = smp.losses.DiceLoss(mode="binary", ignore_index=IGNORE_INDEX)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.val_metrics = SegmentationMetrics()
        self.test_metrics = SegmentationMetrics()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # segmentation_models_pytorch ships no type stubs.
        return cast(torch.Tensor, self.model(image))

    def _compute_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Dice loss's own ignore_index handles masking internally.
        # smp ships no type stubs, hence the cast.
        dice = cast(torch.Tensor, self.dice_loss(logits, mask.unsqueeze(1).float()))
        # BCEWithLogitsLoss has no ignore_index -- mask manually, same
        # pattern as the M0 smoke test and the datamodule's own conventions.
        valid = mask != IGNORE_INDEX
        if not bool(valid.any()):
            # A patch that's entirely missing-data pixels (rare, but real
            # -- e.g. a sensor gap at a tile edge). BCEWithLogitsLoss's
            # mean reduction divides by zero valid elements and silently
            # returns NaN, which would poison every step after it in a
            # long unattended training run. Caught by test_unet.py before
            # this ever ran against real data. Dice's own ignore_index
            # already handles this case correctly (returns 0), so just
            # skip the BCE term rather than let it contribute NaN.
            return dice
        bce = cast(
            torch.Tensor, self.bce_loss(logits.squeeze(1)[valid], mask[valid].float())
        )
        return dice + bce

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self._compute_loss(logits, batch["mask"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self._compute_loss(logits, batch["mask"])
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        self.val_metrics.update(preds, batch["mask"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self) -> None:
        for name, value in self.val_metrics.compute().items():
            if name != "n_pixels":
                self.log(f"val_{name}", value)
        self.val_metrics = SegmentationMetrics()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(batch["image"])
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        self.test_metrics.update(preds, batch["mask"])

    def on_test_epoch_end(self) -> None:
        for name, value in self.test_metrics.compute().items():
            if name != "n_pixels":
                self.log(f"test_{name}", value)
        self.test_metrics = SegmentationMetrics()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class QualitativePanelLogger(pl.Callback):
    """Logs a few VV / prediction / ground-truth panels to W&B at the end
    of each validation epoch -- the "qualitative prediction panels each
    epoch" issue #8 asks for. No-ops gracefully if the trainer isn't
    using a WandbLogger (e.g. local structural testing without wandb
    configured), rather than failing training over a missing logger.
    """

    def __init__(self, n_samples: int = 4) -> None:
        self.n_samples = n_samples

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        if not hasattr(experiment, "log"):  # duck-types the wandb.Run interface
            return
        if trainer.val_dataloaders is None:
            return

        batch = next(iter(trainer.val_dataloaders))
        image = batch["image"][: self.n_samples].to(pl_module.device)
        mask = batch["mask"][: self.n_samples]
        with torch.no_grad():
            pred = (torch.sigmoid(pl_module(image)).squeeze(1) > 0.5).long().cpu()

        panels = [
            wandb.Image(
                image[i, 0].cpu().numpy(),  # VV channel as the base image
                masks={
                    "prediction": {
                        "mask_data": pred[i].numpy(),
                        "class_labels": {0: "not flood", 1: "flood"},
                    },
                    "ground_truth": {
                        "mask_data": mask[i].numpy(),
                        "class_labels": {0: "not flood", 1: "flood", 255: "ignore"},
                    },
                },
            )
            for i in range(min(self.n_samples, image.shape[0]))
        ]
        experiment.log({"val_predictions": panels, "epoch": trainer.current_epoch})
