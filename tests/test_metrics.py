"""SegmentationMetrics tests. Synthetic tensors only -- no real data
needed, so these run in CI.
"""

import math

import torch

from waterline.eval.metrics import SegmentationMetrics


def test_perfect_prediction() -> None:
    target = torch.tensor([0, 0, 1, 1, 0, 1])
    pred = target.clone()
    m = SegmentationMetrics()
    m.update(pred, target)
    result = m.compute()
    assert result["iou"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_all_wrong_on_positives() -> None:
    target = torch.tensor([1, 1, 1, 1])
    pred = torch.tensor([0, 0, 0, 0])
    m = SegmentationMetrics()
    m.update(pred, target)
    result = m.compute()
    assert result["iou"] == 0.0
    assert result["recall"] == 0.0
    assert math.isnan(result["precision"])  # no positive predictions at all


def test_known_confusion_matrix() -> None:
    # 2 TP, 1 FP, 1 FN, 2 TN -- worked out by hand
    target = torch.tensor([1, 1, 0, 1, 0, 0])
    pred = torch.tensor([1, 1, 1, 0, 0, 0])
    m = SegmentationMetrics()
    m.update(pred, target)
    t = m.totals
    assert (t.true_positive, t.false_positive, t.false_negative, t.true_negative) == (
        2,
        1,
        1,
        2,
    )
    assert m.compute()["iou"] == 2 / 4  # TP / (TP+FP+FN)


def test_ignore_index_excluded_from_everything() -> None:
    """A pixel marked ignore_index must not count as either class."""
    target = torch.tensor([1, 1, 255, 255])
    pred = torch.tensor([1, 0, 1, 0])  # predictions on ignored pixels: irrelevant
    m = SegmentationMetrics()
    m.update(pred, target)
    result = m.compute()
    assert result["n_pixels"] == 2  # only the two non-ignored pixels counted
    assert result["recall"] == 0.5  # 1 TP out of 2 real positives


def test_accumulates_across_multiple_updates() -> None:
    """The whole point of the accumulator: counts add up correctly
    across many update() calls, not just the last one.
    """
    m = SegmentationMetrics()
    m.update(torch.tensor([1, 1]), torch.tensor([1, 1]))  # 2 TP
    m.update(torch.tensor([0, 0]), torch.tensor([1, 1]))  # 2 FN
    result = m.compute()
    assert result["n_pixels"] == 4
    assert result["recall"] == 0.5  # 2 TP out of 4 real positives total


def test_handles_batched_and_2d_shapes() -> None:
    """update() must work on (H, W) single patches and (B, H, W) batches
    identically -- flattening shouldn't care which shape it started as.
    """
    target_2d = torch.tensor([[1, 0], [0, 1]])
    pred_2d = torch.tensor([[1, 0], [0, 1]])
    m1 = SegmentationMetrics()
    m1.update(pred_2d, target_2d)

    target_3d = target_2d.unsqueeze(0)
    pred_3d = pred_2d.unsqueeze(0)
    m2 = SegmentationMetrics()
    m2.update(pred_3d, target_3d)

    assert m1.compute() == m2.compute()
