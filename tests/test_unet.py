"""FloodUNet tests. Synthetic tensors, encoder_weights=None throughout
(no network access, no pretrained-weight download) -- these run in CI.
"""

import torch

from waterline.data.mmflood import IGNORE_INDEX
from waterline.models.unet import IN_CHANNELS, FloodUNet


def _tiny_model() -> FloodUNet:
    # resnet18 (not the real resnet34 baseline) -- this only needs to be
    # structurally correct and fast, not the actual baseline's accuracy.
    return FloodUNet(encoder_name="resnet18", encoder_weights=None)


def test_output_shape_matches_input() -> None:
    model = _tiny_model()
    image = torch.randn(2, IN_CHANNELS, 64, 64)
    logits = model(image)
    assert logits.shape == (2, 1, 64, 64)


def test_two_channel_input_only() -> None:
    """VV+VH baseline, not VV/VH/DEM -- see module docstring for why."""
    assert IN_CHANNELS == 2


def test_loss_excludes_ignore_index_pixels() -> None:
    """A target that's entirely ignore_index must not produce a NaN or
    crash -- the BCE term has no valid pixels to compute over.
    """
    model = _tiny_model()
    image = torch.randn(1, IN_CHANNELS, 64, 64)
    logits = model(image)
    mask_all_ignored = torch.full((1, 64, 64), IGNORE_INDEX, dtype=torch.long)
    loss = model._compute_loss(logits, mask_all_ignored)
    # Dice's own ignore_index handling still produces a real number even
    # when BCE's valid-pixel selection is empty; just confirm it doesn't
    # crash and produces a finite value.
    assert torch.isfinite(loss)


def test_loss_decreases_over_a_few_steps() -> None:
    """Not a full training run -- just confirms gradients actually flow
    and the masked loss is learnable, the same property verified against
    real data before this test was written (see PR description).
    """
    model = _tiny_model()
    model.train()
    image = torch.randn(2, IN_CHANNELS, 32, 32)
    mask = torch.randint(0, 2, (2, 32, 32))
    optimizer = model.configure_optimizers()

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(image)
        loss = model._compute_loss(logits, mask)
        # torch's own stub gap on Tensor.backward, hence the ignore.
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
