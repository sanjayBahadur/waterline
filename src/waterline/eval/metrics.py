"""Segmentation metrics: IoU, F1, precision, recall.

An accumulator, not a one-shot function -- call update() once per patch
across a full evaluation pass, then compute() once at the end. Counts
(TP/FP/FN/TN) are what get accumulated, not per-patch ratios -- averaging
per-patch IoU values would weight a 10-pixel patch the same as a
100,000-pixel one, which is wrong.

Pure and dataset-agnostic on purpose: identical whether predictions come
from the Otsu threshold baseline or a trained U-Net, so it's written once
and reused for both (see harness.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

IGNORE_INDEX = 255  # MMFlood's sentinel for missing-data pixels; excluded


@dataclass
class MetricTotals:
    """Raw confusion-matrix counts, and the ratios derived from them."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def iou(self) -> float:
        denom = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denom if denom else math.nan

    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else math.nan

    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else math.nan

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return math.nan
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "n_pixels": float(
                self.true_positive
                + self.false_positive
                + self.false_negative
                + self.true_negative
            ),
        }


class SegmentationMetrics:
    """Accumulates confusion-matrix counts across many update() calls.

    Pixels equal to ignore_index in `target` are excluded entirely --
    MMFlood's own convention for missing/corrupted sensor data, not a
    real class (see CLAUDE.md and scripts/inspect_mmflood.py).
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index
        self.totals = MetricTotals()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: same-shape tensors of 0/1 (or bool). Any extra
        leading batch dimension is fine -- flattened internally.
        """
        pred_flat = pred.reshape(-1).bool()
        target_flat = target.reshape(-1)
        valid = target_flat != self.ignore_index
        pred_valid = pred_flat[valid]
        target_valid = target_flat[valid].bool()

        self.totals.true_positive += int((pred_valid & target_valid).sum())
        self.totals.false_positive += int((pred_valid & ~target_valid).sum())
        self.totals.false_negative += int((~pred_valid & target_valid).sum())
        self.totals.true_negative += int((~pred_valid & ~target_valid).sum())

    def compute(self) -> dict[str, float]:
        return self.totals.as_dict()
