# waterline

Flood extent segmentation from Sentinel-1 radar imagery, with terrain priors to cut false positives in mountainous areas.

**Status:** in development. Nothing here is validated yet — results tables below are placeholders.

---

## Why radar

Optical satellites can't see through cloud, and floods arrive with weather. Synthetic aperture radar can, which is why disaster response agencies lean on Sentinel-1 for flood mapping.

The tradeoff is that SAR is harder to work with. Calm water reflects the radar pulse away from the sensor and returns as a dark patch, which is the signal every SAR flood method depends on. Unfortunately, radar shadow on the lee side of steep terrain is also dark. In flat country this barely matters. In mountains it produces false positives everywhere, and it's the main reason SAR flood products carry terrain caveats.

This project asks a narrow question: **does giving the model elevation-derived context measurably reduce those false positives?**

## Approach

Three stages, each one shippable on its own.

1. **Baseline.** U-Net over Sentinel-1 VV/VH backscatter, trained on MMFlood. Held out by flood event rather than randomly — nearby image tiles are strongly correlated, so a random split leaks and inflates scores.
2. **Terrain-aware.** Same architecture, with DEM-derived slope and Height Above Nearest Drainage appended as input channels. Reported as an ablation against the baseline, with a separate breakdown on high-relief scenes.
3. **Out-of-distribution case study.** Inference over the Bhote Koshi / Trishuli corridor in Nepal following the 26 August 2026 event, as a documented failure analysis rather than a benchmark. See [`docs/nepal-case-study.md`](docs/nepal-case-study.md) for why that distinction matters.

## Results

_Placeholder — to be filled once stage 1 completes._

| Model | Input | IoU (event holdout) | IoU (random split) | FP rate, high-relief |
|---|---|---|---|---|
| Otsu threshold | VV | — | — | — |
| U-Net | VV, VH | — | — | — |
| U-Net + terrain | VV, VH, slope, HAND | — | — | — |

Both split types are reported deliberately. The random-split number is the one that looks good; the event-holdout number is the one that reflects how the model would actually behave on a new flood.

## Nepal case study

The August 2026 Rasuwa disaster was a glacier collapse that sent a debris flow roughly 100 km downstream — mud, rock and ice, not standing water. A model trained to find smooth open water is not expected to segment it, and the published reference products (Copernicus EMS activation EMSR927, UNOSAT product 4257) delineate mudflow extent, so scoring against them would measure a category mismatch rather than model quality.

It's included anyway, for two reasons. It's an honest demonstration of where the method's assumptions break, and the pre-event acquisition over the same terrain works as a free negative control: with no flood present, every positive prediction is a false positive by construction, which gives a labelling-free measure of terrain-induced error in exactly the conditions the terrain channels are meant to address.

## Setup

```bash
git clone https://github.com/sanjayBahadur/waterline
cd waterline
uv sync --extra dev        # or: pip install -e ".[dev]"
pre-commit install
```

Training data is not vendored, and this repo does not automate fetching it —
see [`CLAUDE.md`](CLAUDE.md) for why. Stage it manually via TorchGeo:

```python
from torchgeo.datasets import MMFlood

ds = MMFlood(root="./data/mmflood", split="train", include_dem=True, download=True)
```

MMFlood is indexed by real-world coordinates, not list position — see
[`scripts/inspect_mmflood.py`](scripts/inspect_mmflood.py) for how to
correctly pull a single sample out of it. Expected `data/` layout is
documented in [`CLAUDE.md`](CLAUDE.md#data-handling).

Train:

```bash
python -m waterline.train --config-name baseline
python -m waterline.train --config-name terrain
```

Configs are Hydra; override from the command line, e.g. `trainer.max_epochs=50 data.batch_size=8`.

## Layout

```
src/waterline/
  data/        datamodules, band handling, augmentation
  models/      Lightning modules
  eval/        metrics, event-wise splitting, terrain stratification
  inference/   tiled prediction, GeoTIFF writing, ONNX export
configs/       Hydra configs
scripts/       download, feasibility checks, export
app/           serving (Modal + FastAPI) and web frontend
docs/          method notes, case study, model card
tests/
```

## Data

**MMFlood** — 1,748 Sentinel-1 tiles across 95 flood events in 42 countries, with flood delineation derived from Copernicus EMS activations, plus per-tile DEM and partial hydrography coverage. Loaded via TorchGeo.

**Copernicus DEM GLO-30** — elevation, for slope and HAND derivation.

**JRC Global Surface Water** — permanent water mask, used to exclude rivers and lakes from false-positive accounting.

Datasets carry their own licences, separate from this repository's. Check them before redistributing anything derived.

## Limitations

- Trained on C-band Sentinel-1 only; will not transfer to L-band or X-band sensors without retraining.
- Detects open water. Flooding under dense vegetation or inside built-up areas is substantially harder and is not handled well.
- Debris flows, mudflows and sediment deposition are out of scope, as discussed above.
- This is a research and portfolio project. It is not validated for operational use, and it should not inform emergency decisions.

## References

- Montello, Arnaudo & Rossi (2022). MMFlood: A Multimodal Dataset for Flood Delineation From Satellite Imagery. *IEEE Access.*
- Bonafilia et al. (2020). Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1. *CVPR Workshops.*
- Bountos et al. (2023). Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping.
- Stewart et al. (2022). TorchGeo: Deep Learning With Geospatial Data.

## Licence

Code under MIT. Data under the terms of its respective providers.
