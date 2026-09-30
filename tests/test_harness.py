"""Evaluation harness tests against the real, un-vendored MMFlood dataset.

Marked slow and skipped by default -- CI has no access to the real 47GB
dataset. Run locally with: uv run pytest -m slow
"""

from pathlib import Path

import pytest
import torch

from waterline.eval.harness import evaluate_subset
from waterline.models.otsu import otsu_predict

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


@pytest.mark.slow
def test_otsu_baseline_scores_sanely_on_val() -> None:
    """The real classical baseline (#7), not a placeholder predictor --
    scored on val (7 events, fast) rather than test (34 events, ~100s)
    for routine local runs.
    """
    report = evaluate_subset(otsu_predict, DATA_ROOT, subset="val", patch_size=256)
    assert 0.0 <= report.overall["iou"] <= 1.0
    assert (
        report.overall["recall"] > 0.5
    )  # Otsu over-predicts flood; recall should be high


@pytest.mark.slow
def test_otsu_confirms_high_relief_is_harder() -> None:
    """Locks in the core finding this project exists to investigate:
    the classical baseline should do measurably worse in high-relief
    terrain than flat terrain (radar shadow producing false positives).
    Deterministic (Otsu has no randomness, GridGeoSampler tiles exactly
    the same way every run), so this is a real regression check, not a
    flaky one.
    """
    report = evaluate_subset(otsu_predict, DATA_ROOT, subset="val", patch_size=256)
    flat, high_relief = report.per_relief["flat"], report.per_relief["high-relief"]
    assert high_relief["precision"] < flat["precision"], (
        "expected high-relief terrain to show worse precision than flat "
        "terrain (radar shadow -> false positives) -- if this fails, "
        "either the finding genuinely changed or something in the "
        "relief-bucketing/prediction pipeline broke"
    )
