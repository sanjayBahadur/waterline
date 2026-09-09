"""Smoke tests. Replaced by real coverage as modules land."""

import waterline


def test_version_present() -> None:
    assert waterline.__version__


def test_subpackages_import() -> None:
    from waterline import data, eval, inference, models  # noqa: F401
