"""Classical baseline: Otsu thresholding on VV (issue #7).

The operational method the learned model has to beat to justify itself.
Not a trained model -- Otsu's method looks at one image's own brightness
histogram and finds the threshold that best separates it into two
groups (minimizing within-group variance). Applied per-patch, matching
how it's actually used in the SAR flood-mapping literature: a "global"
Otsu threshold fixed across all images would defeat the point of an
auto-thresholding method built to adapt to each scene's own
calibration.

Uses scikit-image's threshold_otsu -- the standard reference
implementation -- rather than reimplementing a textbook algorithm.

Water is dark (low VV backscatter, see CLAUDE.md's domain primer), so
the predicted flood class is whichever side of the threshold is darker,
not just "below the threshold" -- Otsu's method only finds a split
point, it doesn't know which side means "water."
"""

from __future__ import annotations

import torch
from skimage.filters import threshold_otsu


def otsu_predict(image: torch.Tensor) -> torch.Tensor:
    """Predict flood (1) / not-flood (0) for one image patch via Otsu
    thresholding on its VV channel.

    Args:
        image: (C, H, W), VV as channel 0 (see CLAUDE.md's channel-order
            rule). Other channels are ignored -- this baseline is VV-only
            by definition.

    Returns:
        (H, W) long tensor of 0/1 predictions.
    """
    vv = image[0].numpy()

    if vv.min() == vv.max():
        # A uniform patch has no histogram to split -- Otsu is undefined.
        # Uniform backscatter this flat is far more consistent with a
        # calm, uniform surface (water) than textured land, so predict
        # flood everywhere rather than crash or default to "nothing."
        return torch.ones(vv.shape, dtype=torch.long)

    # scikit-image ships no type stubs, hence the ignore below.
    threshold: float = threshold_otsu(vv)  # type: ignore[no-untyped-call]

    # Water is dark: the class below the threshold is flood, by domain
    # knowledge, not because Otsu says so -- it only finds a split
    # point, it has no notion of which side means what.
    pred = vv <= threshold

    return torch.from_numpy(pred.astype("int64"))
