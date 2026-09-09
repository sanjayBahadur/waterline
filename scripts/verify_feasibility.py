#!/usr/bin/env python3
"""
Feasibility verification for the SAR flood segmentation project.

Two checks:
  CHECK 1 - Does Sentinel-1 have usable pre/post acquisitions over the
            Rasuwa / Bhote Koshi / Trishuli corridor bracketing 26 Aug 2026,
            on a MATCHING relative orbit? (Same orbit matters: change
            detection across different viewing geometries in steep terrain
            is mostly noise.)

  CHECK 2 - Can the training datasets actually be pulled?

Install:
    pip install pystac-client planetary-computer odc-stac rich

Run:
    python verify_feasibility.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

# ----------------------------------------------------------------------------
# Area of interest: Rasuwagadhi/Timure -> Syabrubesi -> Betrawati -> Trishuli
# Also covers the reported barrier lake near 85.510E, 28.292N
# ----------------------------------------------------------------------------
AOI_BBOX = [85.05, 27.75, 85.62, 28.42]
EVENT_DATE = datetime(2026, 8, 26)
DATE_RANGE = "2026-07-15/2026-09-30"  # wide enough to see the full revisit pattern

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
AWS_STAC = "https://earth-search.aws.element84.com/v1"


def check_sentinel1(stac_url: str, collection: str, label: str, sign: bool = False):
    """Query a STAC API for Sentinel-1 scenes over the AOI and group by orbit."""
    import pystac_client

    print(f"\n{'=' * 70}")
    print(f"  {label}  ->  collection '{collection}'")
    print(f"{'=' * 70}")

    try:
        if sign:
            import planetary_computer

            catalog = pystac_client.Client.open(
                stac_url, modifier=planetary_computer.sign_inplace
            )
        else:
            catalog = pystac_client.Client.open(stac_url)

        search = catalog.search(
            collections=[collection],
            bbox=AOI_BBOX,
            datetime=DATE_RANGE,
        )
        items = list(search.items())
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None

    if not items:
        print("  No items returned. Collection may be unavailable or AOI uncovered.")
        return None

    print(f"  {len(items)} scenes found\n")

    # Group by relative orbit + pass direction. A usable change-detection pair
    # needs one pre-event and one post-event scene in the SAME group.
    groups: dict[tuple, list] = defaultdict(list)
    for it in items:
        p = it.properties
        orbit = (
            p.get("sat:relative_orbit")
            or p.get("sar:relative_orbit")
            or p.get("s1:orbit_source")
            or "unknown"
        )
        direction = p.get("sat:orbit_state", "?")
        groups[(orbit, direction)].append(it)

    usable_pairs = 0
    for (orbit, direction), its in sorted(groups.items(), key=lambda x: str(x[0])):
        its.sort(key=lambda i: i.datetime)
        pre = [i for i in its if i.datetime.replace(tzinfo=None) < EVENT_DATE]
        post = [i for i in its if i.datetime.replace(tzinfo=None) >= EVENT_DATE]

        flag = ""
        if pre and post:
            usable_pairs += 1
            gap_pre = (EVENT_DATE - pre[-1].datetime.replace(tzinfo=None)).days
            gap_post = (post[0].datetime.replace(tzinfo=None) - EVENT_DATE).days
            flag = f"  <<< USABLE PAIR (pre -{gap_pre}d / post +{gap_post}d)"

        print(f"  orbit {orbit} [{direction}]  {len(pre)} pre / {len(post)} post{flag}")
        for i in its:
            marker = "*" if i.datetime.replace(tzinfo=None) >= EVENT_DATE else " "
            mode = i.properties.get("sar:instrument_mode", "?")
            pol = i.properties.get("sar:polarizations", "?")
            print(f"      {marker} {i.datetime:%Y-%m-%d %H:%M}  {mode}  {pol}")

    print(f"\n  >>> VERDICT: {usable_pairs} same-orbit pre/post pair(s) available")
    if usable_pairs == 0:
        print("  >>> Nepal case study is NOT viable on this collection.")
    return usable_pairs


def check_datasets():
    """Confirm the training datasets are reachable through TorchGeo."""
    print(f"\n{'=' * 70}")
    print("  CHECK 2 - training dataset availability")
    print(f"{'=' * 70}")

    try:
        import torchgeo
        from torchgeo.datasets import CopernicusBenchFloodS1, MMFlood  # noqa: F401

        print(f"  torchgeo {torchgeo.__version__} OK")
        print("  MMFlood                 available (S1 + DEM + hydro, EMS labels)")
        print("  CopernicusBenchFloodS1  available (Kuro Siwo subset, 3-class)")
    except ImportError as e:
        print(f"  torchgeo import failed: {e}")
        print("  -> pip install torchgeo")
        return

    print("\n  To pull MMFlood (check free disk first - it is 11 tar parts):")
    print("    from torchgeo.datasets import MMFlood")
    print("    ds = MMFlood(root='./data/mmflood', split='train',")
    print("                 include_dem=True, download=True)")
    print("    print(len(ds)); print(ds[0]['image'].shape)")
    print("\n  include_dem=True concatenates DEM after the VV/VH bands, which")
    print("  makes the terrain-aware ablation a one-flag experiment.")


if __name__ == "__main__":
    print("SAR FLOOD PROJECT - FEASIBILITY VERIFICATION")
    print(f"AOI: {AOI_BBOX}   event: {EVENT_DATE:%Y-%m-%d}")

    # GRD is openly accessible; RTC on Planetary Computer needs a free
    # subscription key. Try both, plus the AWS mirror.
    check_sentinel1(PC_STAC, "sentinel-1-grd", "CHECK 1a - Planetary Computer GRD")
    check_sentinel1(
        PC_STAC, "sentinel-1-rtc", "CHECK 1b - Planetary Computer RTC", sign=True
    )
    check_sentinel1(AWS_STAC, "sentinel-1-grd", "CHECK 1c - AWS Earth Search GRD")

    check_datasets()

    print("\n" + "=" * 70)
    print("  Decision rule:")
    print("   - >=1 usable pair on RTC  -> Nepal case study is GO")
    print("   - pairs only on GRD       -> GO, but budget time for terrain")
    print("                                correction via ASF HyP3")
    print("   - no pairs at all         -> drop to Tier 1+2, project still fine")
    print("=" * 70)
