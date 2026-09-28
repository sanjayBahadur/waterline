"""Otsu baseline tests. Synthetic tensors only -- no real data needed."""

import torch

from waterline.models.otsu import otsu_predict


def test_predicts_darker_region_as_flood() -> None:
    """A clearly bimodal image: a dark square (water) in a bright field
    (land). Otsu should find the split and flag the dark region.
    """
    vv = torch.full((20, 20), 10.0)
    vv[5:15, 5:15] = 0.1  # dark square in the middle
    image = vv.unsqueeze(0)  # (1, 20, 20) -- single-channel is fine, only VV used

    pred = otsu_predict(image)
    assert pred.shape == (20, 20)
    assert pred.dtype == torch.long
    # The dark square should be predicted flood (1); the bright field not.
    assert pred[5:15, 5:15].float().mean() > 0.9
    assert pred[0:5, 0:5].float().mean() < 0.1


def test_uniform_patch_does_not_crash() -> None:
    """Otsu's threshold is undefined on a flat histogram -- must not
    raise, and the documented fallback (predict flood) must hold.
    """
    image = torch.full((1, 10, 10), 3.0)
    pred = otsu_predict(image)
    assert pred.shape == (10, 10)
    assert bool((pred == 1).all())


def test_only_first_channel_used() -> None:
    """VV-only baseline by definition -- extra channels (VH, DEM) must
    not influence the prediction.
    """
    vv = torch.full((10, 10), 10.0)
    vv[0:5, :] = 0.1
    other_channels = torch.rand(2, 10, 10) * 1000  # deliberately huge, irrelevant noise
    image = torch.cat([vv.unsqueeze(0), other_channels], dim=0)

    pred = otsu_predict(image)
    assert pred[0:5, :].float().mean() > 0.9
    assert pred[5:10, :].float().mean() < 0.1
