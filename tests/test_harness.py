"""Evaluation harness tests against the real, un-vendored MMFlood dataset.

Marked slow and skipped by default -- CI has no access to the real 47GB
dataset. Run locally with: uv run pytest -m slow
"""

from pathlib import Path

import pytest
import torch

from waterline.eval.harness import evaluate_subset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "mmflood"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="MMFlood not staged locally (see CLAUDE.md)"
)


def _dark_pixel_predict(image: torch.Tensor) -> torch.Tensor:
    """Trivial "water is dark" rule -- not a real baseline (that's Otsu,
    issue #7), just enough signal to exercise the harness end-to-end.
    """
    vv = image[0]
    return (vv < 0.05).long()


@pytest.mark.slow
def test_evaluate_val_split_runs_and_reports_all_seven_events() -> None:
    report = evaluate_subset(
        _dark_pixel_predict, DATA_ROOT, subset="val", patch_size=256
    )
    assert len(report.per_event) == 7  # val is 7 events, verified against metadata
    assert report.overall["n_pixels"] > 0
    for metrics in report.per_event.values():
        assert 0.0 <= metrics["iou"] <= 1.0


@pytest.mark.slow
def test_relief_buckets_both_populated() -> None:
    """Both buckets should get real pixels on the val split -- if one
    comes back empty, the relief threshold or DEM channel handling is
    broken, not just "this split happens to be all flat."
    """
    report = evaluate_subset(
        _dark_pixel_predict, DATA_ROOT, subset="val", patch_size=256
    )
    assert report.per_relief["flat"]["n_pixels"] > 0
    assert report.per_relief["high-relief"]["n_pixels"] > 0
