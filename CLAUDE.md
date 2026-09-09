# CLAUDE.md

Operating context for `waterline`. Read this fully before proposing work.

---

## What this is

Flood extent segmentation from Sentinel-1 SAR imagery, with terrain-derived
input channels to reduce false positives in mountainous areas. Solo portfolio
project targeting new-grad ML/CV roles, with a live interactive web demo as the
primary artifact.

**The author must be able to defend every technical decision in an interview.**
This constraint outranks velocity. See "Working agreement" below — it changes
how you should behave.

---

## How to talk to me

Two different registers. Do not mix them up.

**In chat: plain English.** The author is building expertise as the project
proceeds and is not yet fluent in remote sensing or deep learning vocabulary.

**In the repo: professional.** Code comments, commit messages, PR descriptions,
docstrings and documentation use correct technical terminology, because
recruiters and engineers read those. Never dumb down repo artifacts.

So: explain it simply to me, then write it properly in the code.

### Rules for chat

- **Define jargon the first time it appears in a session.** One short clause is
  enough — "dice loss (a scoring method that handles the case where the thing
  you're looking for covers only a small fraction of the image)".
- **Never present a decision using terms I haven't been given.** If a choice
  depends on understanding a concept, explain the concept first, then ask.
- **Lead with what and why, follow with how.** Say what you're about to do in
  one plain sentence before showing code.
- **After finishing, report in plain language:** what changed, why it changed,
  what it means, and what happens next. Not a diff summary — an explanation.
- **State the stakes of a decision.** Say whether a choice is easily reversed or
  hard to undo later, so I know how much attention it deserves.
- **Say when something is uncertain**, and say what would resolve it.
- **No walls of text.** Short paragraphs. Get to the point.
- **Never assume I understood.** If something was explained several sessions
  ago, briefly re-anchor it rather than assuming it stuck.
- If I ask what something means, answer directly without making me feel slow
  for asking. Curiosity is the point of this project.

### When presenting a decision, use this shape

```
What's being decided:  <one plain sentence>
Option A:              <plain description> — good because X, costs you Y
Option B:              <plain description> — good because X, costs you Y
My recommendation:     <which, and the single most important reason>
Reversible?            <easy to change later, or locked in>
```

Then wait. Do not implement while asking.

### Reporting completed work

```
What I did:      <plain sentence>
Why that way:    <the reasoning, in plain terms>
What it means:   <what is now true that wasn't before>
Watch out for:   <anything fragile, unverified, or assumed>
Next:            <the immediate next step>
```

### Interview readiness

Roughly once per milestone, ask me to explain a decision back in my own words.
If I can't, that part isn't finished — regardless of whether the code works.
This is not optional politeness; it is the primary success criterion for the
project.

---

## Domain primer

Read this before touching modeling code. Most bugs here come from treating SAR
like RGB.

**Why radar.** Optical satellites cannot see through cloud, and floods arrive
with storms. SAR is active — it emits a pulse and measures the echo — so it
works through cloud and at night. This is why operational flood response uses
Sentinel-1.

**The core signal.** Calm open water reflects the radar pulse away from the
sensor (specular reflection), so water returns very low backscatter and appears
dark. Rough surfaces scatter energy back and appear bright. Every SAR flood
method depends on "water is dark."

**Why mountains break it.** SAR is side-looking. Steep slopes facing away from
the sensor sit in radar shadow, which is also dark. A model trained on flat
terrain will classify shadowed valley walls as flood. Layover and foreshortening
compound this. Terrain-induced false positives are the central problem this
project addresses.

**Polarization.** VV and VH are different transmit/receive polarizations.
VV is generally more sensitive to open water; VH carries more information about
volume scattering (vegetation). Both are used.

**Units.** Backscatter is typically converted to decibels. Do not apply RGB
image conventions — no colour jitter, no ImageNet normalization statistics. Use
dataset-derived per-channel statistics.

**Speckle.** SAR has multiplicative speckle noise, not additive Gaussian.
Standard denoising assumptions do not transfer.

---

## Finalized decisions

These are settled. Do not re-litigate them unless new evidence appears; if it
does, say so explicitly rather than quietly changing course.

| Decision | Choice | Reason |
|---|---|---|
| Framework | PyTorch + Lightning | Entire EO ecosystem is PyTorch. TensorFlow is rejected. |
| Geo library | TorchGeo | Handles CRS, resolution, multiband, geo-aware samplers |
| Primary dataset | **MMFlood** | 1,748 S1 tiles, 95 events, 42 countries, EMS-derived labels, **ships DEM** |
| Secondary dataset | CopernicusBenchFloodS1 | Kuro Siwo subset, 3-class, includes pre-event imagery |
| Architectures | U-Net, then SegFormer | Two, honestly compared. Not five. |
| Classical baseline | Otsu threshold on VV | The operational method the model must beat |
| Split strategy | Event-wise holdout | Random splits leak via spatial autocorrelation |
| Serving | Modal (serverless GPU) | Scale-to-zero, per-second billing. AWS rejected — no cheap scale-to-zero GPU. |
| Frontend | React + MapLibre GL | Static hosting on Vercel/Cloudflare, free tier |
| Config | Hydra | No hardcoded hyperparameters |
| Tracking | Weights & Biases | |
| Analytics | DuckDB spatial | Demonstrates SQL alongside Python |

### Explicitly rejected, with reasons

- **TensorFlow** — unused in Earth observation; splitting focus produces two
  shallow competencies instead of one deep one.
- **Sen1Floods11** — only 446 hand-labeled chips, no DEM, not in TorchGeo.
  MMFlood supersedes it. Cite it as related work only.
- **Geospatial foundation models (Prithvi, Clay, TerraMind)** — pretrained on
  optical bands; they do not cleanly apply to SAR-only input. Knowing this is a
  talking point, not an omission.
- **AWS SageMaker** — serverless inference is CPU-only; GPU endpoints bill
  continuously.
- **Change-detection reformulation for Nepal** — technically correct and worth
  mentioning verbally, but a different system. Out of scope for v1.
- **Parallel multi-agent implementation** — review bandwidth is the bottleneck,
  not code generation.

---

## Project structure

```
src/waterline/
  data/        datamodules, band handling, augmentation, splits
  models/      Lightning modules
  eval/        metrics, event-wise splitting, terrain stratification
  inference/   tiled prediction, COG writing, ONNX export
configs/       Hydra configs
scripts/       feasibility checks, export, one-off utilities
app/
  backend/     Modal + FastAPI
  frontend/    React + MapLibre
docs/          method notes, Nepal case study, model card
tests/
notebooks/     exploration only; nbstripout enforced
```

---

## Data handling

**The author manages all dataset acquisition and placement manually.** Do not
write download automation, do not attempt network fetches of datasets, and do
not assume data is present.

Expected layout (confirm before use, do not create):

```
data/
  mmflood/          MMFlood via TorchGeo, include_dem=True
  copernicus_bench/ CopernicusBenchFloodS1
  nepal/            Sentinel-1 scenes over Bhote Koshi / Trishuli corridor
  aux/              JRC Global Surface Water, OSM extracts
```

`data/` and all checkpoints are gitignored. If code needs data that is absent,
fail loudly with a clear message naming the expected path — never silently
generate synthetic substitutes.

---

## Hard domain rules

Violating any of these invalidates results.

1. **Splits are grouped by flood event, never random.** Adjacent tiles are
   spatially correlated. Both numbers get reported — the honest event-holdout
   score and the inflated random-split score — and the gap is a headline
   finding, not an embarrassment.
2. **Channel order is VV, VH, then optional terrain channels** (slope, HAND).
   Document any deviation at the call site.
3. **Georeferenced outputs must preserve CRS.** Verify in QGIS, not by eye.
   Output Cloud-Optimized GeoTIFF.
4. **Permanent water is masked from false-positive accounting** using JRC Global
   Surface Water. Rivers and lakes are not floods.
5. **The terrain ablation must be a clean ablation** — identical architecture,
   hyperparameters and seeds. Only the input channels change.
6. **Never report a metric without stating which split produced it.**

---

## Nepal case study — constraints

The 26 August 2026 Rasuwa event was a glacier collapse producing a debris flow
(mud, rock, ice) that traveled ~100 km down the Bhote Koshi and Trishuli.

**It is a qualitative case study, not a benchmark.** The reasons matter:

- A model trained on smooth open water is not expected to segment a rough
  debris flow. Rough surfaces return bright, not dark.
- Published references — Copernicus EMS activation **EMSR927** and UNOSAT
  product **4257** — delineate *mudflow/rockflow extent*, not water. Scoring a
  water segmenter against them measures a category mismatch, not model quality.
- **Do not compute or report IoU against these products.** If asked to, refuse
  and explain why.

**The quantitative component is the pre-event negative control.** Run inference
on a pre-event Sentinel-1 acquisition over the same terrain. Mask permanent
water. With no flood present, every remaining positive is a false positive by
construction. Measure with and without terrain channels. This needs no hand
labeling and directly tests the M2 hypothesis.

**Tone.** Hundreds of people died. Write soberly. Frame strictly as post-event
mapping. Never imply operational, predictive, or early-warning value. Cite
humanitarian sources. State limitations plainly.

---

## Milestones

- **M0 Feasibility** — S1 coverage check over Rasuwa (same relative orbit pre/post
  pairs required), MMFlood load, single-batch overfit smoke test
- **M1 Baseline** — scaffold, CI, datamodule with event-wise splits, Otsu
  baseline, U-Net, evaluation harness
- **M2 Terrain** — slope and HAND derivation, terrain ablation, SegFormer
  comparison. **This produces the headline result.**
- **M3 Serving** — tiled inference, COG output, ONNX export, Modal endpoint
- **M4 Demo** — map with pre/post swipe, impact analytics panel, live STAC tab
- **M5 Case study** — Nepal (gated on M0; falls back to a mountainous European
  event already in MMFlood if coverage fails)
- **M6 Release** — model card, reproducibility pass, video, v1.0.0

Effort allocation is roughly 40% modeling, 60% deployment/frontend/writing.
This is deliberate: marginal IoU is worth little; a working public demo is rare.

---

## Engineering conventions

- Python 3.12, `src/` layout, ruff + mypy (strict), pre-commit hooks
- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
- One issue per PR. Squash merge. Link with `Closes #N`.
- Branch protection on `main`: no direct pushes, CI must pass
- CI under 5 minutes: lint, types, tests, plus a smoke training run on a fixture
  subset so broken training fails in CI rather than on a rented GPU
- Small PRs. If a diff exceeds ~400 lines, split it.
- Tests for: dataloader shapes and CRS, split-group disjointness, tiling
  reassembly correctness, ONNX/PyTorch numerical parity

---

## GitHub workflow

The GitHub connector is available. Read issues, milestones and labels from the
repo rather than inferring the plan. Before starting work:

1. Read the issue and its acceptance criteria
2. Restate the plan and wait for approval before implementing
3. Branch as `feat/<issue-number>-<short-slug>`
4. Open a PR referencing the issue

Do not close issues, edit milestones, or merge PRs autonomously.

---

## Working agreement

**Ask before deciding.** These are design decisions the author must own and
defend. Present options with tradeoffs; do not choose unilaterally:

- Loss functions and class weighting
- Evaluation metrics and split strategy
- Architecture and encoder choices
- Normalization and augmentation strategy
- Anything affecting reported numbers

**Explain, don't just implement.** When writing non-obvious code, state why in
a comment or the PR description. The author has to be able to reconstruct the
reasoning months later without you.

**Surface uncertainty.** If something is unverified — a dataset property, a
library behavior, an assumption about the data — say so rather than asserting.
Wrong confident claims are worse than flagged unknowns here.

**Prefer the boring solution.** This repo is read by humans evaluating judgment.
Clever abstractions with one call site are a liability.

**Negative results are results.** If the terrain channels do not help, if the
deep model does not beat Otsu, if the Nepal inference is uninterpretable —
report it clearly. Honest negative findings are more defensible than tuned
positive ones, and the author has been advised to present them as such.

---

## Glossary

Shared reference. When one of these comes up in chat, give the plain reading
first. Use the precise term in code and documentation.

**SAR / synthetic aperture radar** — a satellite that emits radio pulses and
measures the echo, instead of taking a photograph. Works through cloud and at
night.

**Backscatter** — how much of the pulse bounced back. Low means dark, high
means bright.

**VV / VH** — two ways the radar sends and listens. Think of them as two
different channels of the same scene, like the red and green channels of a
photo but with different physical meaning.

**Speckle** — the grainy salt-and-pepper texture in radar images. A property
of how radar works, not sensor noise, and it does not behave like the noise
image models usually assume.

**dB / decibels** — the scale backscatter is usually expressed in. Compresses a
very wide range of values into a manageable one.

**Segmentation** — labelling every pixel in an image. Here: flood or not flood.

**U-Net** — a standard architecture for segmentation. Compresses the image
down to understand context, then expands back up to produce a pixel-level
answer.

**IoU (intersection over union)** — the standard score for segmentation.
Overlap between prediction and truth, divided by their combined area. 1.0 is
perfect, 0 is no overlap.

**Class imbalance** — when the thing you're detecting covers a small fraction
of the image. Flood pixels are rare, so a model that predicts "no flood"
everywhere scores well on plain accuracy while being useless. Loss choice
compensates for this.

**Ablation** — changing exactly one thing and re-measuring, to prove that one
thing caused the difference.

**Holdout / split** — the data reserved for testing, never trained on. Here it
is split by flood event, not randomly.

**Spatial autocorrelation** — nearby places resemble each other. This is why a
random split cheats: neighbouring tiles end up on both sides of the fence.

**DEM (digital elevation model)** — a map of ground height.

**Slope** — steepness computed from the DEM.

**HAND (height above nearest drainage)** — how far above the nearest stream a
point sits. Water collects in low places, so this is a strong hint about where
flooding is physically possible.

**Radar shadow** — the dark area behind a steep slope where the pulse never
reached. Looks like water to a naive model. The core problem this project
addresses.

**RTC (radiometric terrain correction)** — preprocessing that compensates for
terrain distorting the radar signal.

**GRD** — a less-processed Sentinel-1 product. Needs terrain correction applied
before use in mountains.

**Relative orbit** — which repeating track the satellite was on. Two images of
the same place from different tracks were taken from different angles, so
comparing them directly is unreliable.

**STAC** — a standard catalogue format for searching satellite imagery online,
so you can query for scenes instead of downloading archives.

**COG (cloud-optimized GeoTIFF)** — a georeferenced image file that can be read
in pieces over the network. The standard output format here.

**CRS (coordinate reference system)** — how pixel positions map to real
locations on Earth. Getting this wrong silently puts your results in the wrong
place.

**ONNX** — a portable format for a trained model, so it can run without
PyTorch installed.

**Scale-to-zero** — hosting that costs nothing when nobody is using it.

---

## Reference

- MMFlood — Montello, Arnaudo & Rossi (2022), *IEEE Access*, doi 10.1109/ACCESS.2022.3205419
- Kuro Siwo — Bountos et al. (2023)
- Sen1Floods11 — Bonafilia et al. (2020), CVPR Workshops (related work only)
- TorchGeo — Stewart et al. (2022)
- Copernicus EMS EMSR927 — Nepal activation, 26 Aug 2026
- UNOSAT product 4257 — mudflow extent, Sentinel-2, 27 Aug 2026
