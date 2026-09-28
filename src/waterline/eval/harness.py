"""Evaluation harness: run a prediction function over MMFlood events,
broken down overall, per flood event, and per terrain-relief bucket.

Loops one event at a time rather than sampling from the pooled test set,
specifically so each patch's originating event is known -- torchgeo's
sample dict doesn't carry that back on its own (checked the library's
documented Sample keys -- image/mask/label/bbox_xyxy/prediction -- before
assuming otherwise; there's no bounds/source-file key to recover it from).

Terrain-relief bucketing is a placeholder, not the real M2 measure: it
uses DEM standard deviation within each patch as a cheap proxy for "how
mountainous is this," since true slope/HAND derivation is M2's job. The
default threshold (8m) is grounded in the real training data's own
distribution -- median ~7.5m std-dev over 256x256 patches, sampled from
data/mmflood on 2026-09-28 -- not picked arbitrarily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torchgeo.samplers import GridGeoSampler

from waterline.data.mmflood import event_ids_for_subset, split_for_events
from waterline.eval.metrics import SegmentationMetrics

ReliefBucket = Literal["flat", "high-relief"]
RELIEF_STD_THRESHOLD_M = 8.0  # see module docstring

#: (channels, height, width) image patch -> (height, width) 0/1 prediction
PredictFn = Callable[[torch.Tensor], torch.Tensor]


def relief_bucket(
    dem: torch.Tensor, threshold: float = RELIEF_STD_THRESHOLD_M
) -> ReliefBucket:
    """Placeholder terrain stratification -- see module docstring."""
    return "high-relief" if dem.std().item() >= threshold else "flat"


@dataclass
class EvaluationReport:
    overall: dict[str, float]
    per_event: dict[str, dict[str, float]]
    per_relief: dict[str, dict[str, float]]


def evaluate(
    predict_fn: PredictFn,
    root: Path,
    event_ids: list[str],
    include_dem: bool = True,
    patch_size: int = 256,
    relief_threshold: float = RELIEF_STD_THRESHOLD_M,
) -> EvaluationReport:
    """Score predict_fn over the given events, one event at a time.

    Uses a GridGeoSampler (deterministic, non-overlapping) so every pixel
    in every event is scored exactly once -- unlike training, evaluation
    must not depend on which random patches happened to get drawn.

    Args:
        predict_fn: takes one (C, H, W) image patch, returns a (H, W)
            0/1 (or bool) prediction. Works identically for a threshold
            rule (Otsu) or a trained model's forward pass.
        root: MMFlood dataset root (e.g. data/mmflood).
        event_ids: which events to evaluate -- e.g.
            event_ids_for_subset(root, "test").
        include_dem: must match how predict_fn expects its input built;
            also required for the relief breakdown (DEM is the last
            channel).
        patch_size: evaluation patch size, in pixels.
        relief_threshold: DEM std-dev cutoff between "flat" and
            "high-relief" -- see module docstring.
    """
    overall = SegmentationMetrics()
    per_event: dict[str, dict[str, float]] = {}
    per_relief = {
        "flat": SegmentationMetrics(),
        "high-relief": SegmentationMetrics(),
    }

    for event in event_ids:
        dataset = split_for_events(root, [event], include_dem)
        sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
        event_metrics = SegmentationMetrics()

        for query in sampler:
            sample = dataset[query]
            image, mask = sample["image"], sample["mask"]
            pred = predict_fn(image)

            event_metrics.update(pred, mask)
            overall.update(pred, mask)
            if include_dem:
                bucket = relief_bucket(image[-1], relief_threshold)
                per_relief[bucket].update(pred, mask)

        per_event[event] = event_metrics.compute()

    return EvaluationReport(
        overall=overall.compute(),
        per_event=per_event,
        per_relief={bucket: m.compute() for bucket, m in per_relief.items()},
    )


def evaluate_subset(
    predict_fn: PredictFn,
    root: Path,
    subset: Literal["train", "val", "test"],
    include_dem: bool = True,
    patch_size: int = 256,
    relief_threshold: float = RELIEF_STD_THRESHOLD_M,
) -> EvaluationReport:
    """Convenience wrapper: evaluate over MMFlood's own event-wise subset
    (e.g. "test") directly, without the caller needing to look up event
    IDs themselves first.
    """
    events = event_ids_for_subset(root, subset)
    return evaluate(predict_fn, root, events, include_dem, patch_size, relief_threshold)
