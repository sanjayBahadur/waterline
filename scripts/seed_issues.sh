#!/usr/bin/env bash
# Seed labels, milestones and issues for the waterline repo.
#
#   gh auth login
#   cd <repo>
#   bash seed_issues.sh
#
# Idempotent-ish: label/milestone creation failures are ignored so you can
# re-run after editing. Issues are NOT deduplicated — edit before re-running.

set -uo pipefail

echo "==> labels"
gh label create "area:data"      --color 0E8A16 --description "Datasets, loading, preprocessing"        2>/dev/null
gh label create "area:model"     --color 1D76DB --description "Architectures, training, evaluation"     2>/dev/null
gh label create "area:infra"     --color 5319E7 --description "CI, packaging, tooling"                  2>/dev/null
gh label create "area:serving"   --color B60205 --description "Inference API, deployment"               2>/dev/null
gh label create "area:frontend"  --color FBCA04 --description "Web demo"                                2>/dev/null
gh label create "area:docs"      --color 006B75 --description "Writing, model card, case study"         2>/dev/null
gh label create "type:spike"     --color D4C5F9 --description "Timeboxed investigation, may be dropped" 2>/dev/null
gh label create "blocked"        --color 000000 --description "Waiting on an external answer"           2>/dev/null

echo "==> milestones"
create_ms () {
  gh api "repos/{owner}/{repo}/milestones" -f title="$1" -f description="$2" >/dev/null 2>&1 \
    && echo "    + $1" || echo "    . $1 (exists?)"
}
create_ms "M0 Feasibility"  "Verify the project is buildable before committing to it"
create_ms "M1 Baseline"     "Trained SAR flood segmentation model with honest evaluation"
create_ms "M2 Terrain"      "Terrain-aware variant and measured false-positive reduction"
create_ms "M3 Serving"      "Live inference deployed with scale-to-zero"
create_ms "M4 Demo"         "Interactive web frontend and impact analytics"
create_ms "M5 Case study"   "Nepal out-of-distribution analysis"
create_ms "M6 Release"      "Packaging, docs, video, v1.0.0"

new () { # new <milestone> <labels> <title> <body>
  gh issue create --milestone "$1" --label "$2" --title "$3" --body "$4" >/dev/null \
    && echo "    + $3"
}

echo "==> issues"

# ---------------------------------------------------------------- M0
new "M0 Feasibility" "area:data,type:spike" \
"Verify Sentinel-1 pre/post coverage over Rasuwa corridor" \
"Run \`scripts/verify_feasibility.py\`. Determine whether same-relative-orbit
pre- and post-event acquisitions exist over the Bhote Koshi / Trishuli corridor
bracketing 2026-08-26.

Same orbit matters: cross-geometry comparison in steep terrain is mostly noise.

**Exit criteria:** documented list of candidate scene pairs, or a written
decision that M5 is dropped in favour of a mountainous European event already
present in MMFlood."

new "M0 Feasibility" "area:data,type:spike" \
"Confirm MMFlood downloads and inspect a sample" \
"Pull MMFlood via TorchGeo with \`include_dem=True\`. Record download size and
wall time. Load one sample, confirm channel order and that DEM is concatenated
after VV/VH. Plot a tile with its mask.

**Exit criteria:** notebook or script output showing tensor shapes, value
ranges per channel, and one rendered figure."

new "M0 Feasibility" "area:model,type:spike" \
"End-to-end smoke run: overfit a single batch" \
"Get U-Net + Lightning training on one batch until loss approaches zero.
Nothing about generalisation — this only proves the plumbing is connected.

**Exit criteria:** loss curve showing successful overfit, and a recorded
estimate of seconds/epoch on the full train split."

# ---------------------------------------------------------------- M1
new "M1 Baseline" "area:infra" \
"Repository scaffold and tooling" \
"\`pyproject.toml\` with pinned deps, \`src/\` layout, ruff + mypy, pre-commit
hooks, \`.gitignore\` covering \`data/\` and checkpoints, MIT licence,
\`CITATION.cff\`, PR template.

Branch protection on \`main\`: no direct pushes, require CI to pass."

new "M1 Baseline" "area:infra" \
"CI pipeline" \
"GitHub Actions: lint, type-check, unit tests on push and PR. Cache deps.
Include a 2-minute smoke training run on a fixture subset so broken training
code fails CI rather than failing silently at 2am.

Keep total runtime under 5 minutes."

new "M1 Baseline" "area:data" \
"Data module with event-wise splitting" \
"Lightning DataModule over MMFlood. Splits grouped **by flood event**, never
randomly — adjacent tiles are correlated and a random split leaks.

Expose both split strategies behind a flag so the inflated random-split number
can be reported alongside the honest one.

Include augmentation appropriate to SAR: flips and rotations are fine, colour
jitter is meaningless on backscatter."

new "M1 Baseline" "area:model" \
"Classical baseline: Otsu thresholding on VV" \
"Implement the standard operational method — histogram thresholding on
backscatter — and score it on the same splits.

This is the number the deep model has to beat to justify itself. If it doesn't,
that is a finding worth reporting, not a failure to hide."

new "M1 Baseline" "area:model" \
"U-Net baseline on VV/VH" \
"\`segmentation_models_pytorch\` U-Net, pretrained encoder, first conv adapted
for 2-channel input. Dice + BCE loss; document the class imbalance ratio that
motivates it.

Log to W&B: loss curves, IoU, F1, precision/recall, and qualitative prediction
panels each epoch."

new "M1 Baseline" "area:model" \
"Evaluation harness" \
"IoU, F1, precision, recall, plus per-event breakdown so single-event failures
are visible rather than averaged away.

Stratify by terrain relief using the DEM, since that split is the whole point
of M2."

new "M1 Baseline" "area:docs" \
"Write up baseline results" \
"Fill the README results table. Include the random-vs-event-holdout gap and
explain what it means. Add three qualitative figures: a clear success, a near
miss, and an outright failure."

# ---------------------------------------------------------------- M2
new "M2 Terrain" "area:data" \
"Derive slope and HAND from DEM" \
"Compute slope from the bundled DEM. Compute Height Above Nearest Drainage
(pysheds or richdem). Normalise sensibly and cache to disk — recomputing per
epoch will dominate training time.

Sanity-check both against a known-hilly and a known-flat tile."

new "M2 Terrain" "area:model" \
"Terrain-aware model and ablation" \
"Retrain with slope and HAND as additional input channels. Identical
hyperparameters and seeds — this must be a clean ablation.

Report overall IoU delta and, separately, false-positive rate on high-relief
scenes. The second number is the headline."

new "M2 Terrain" "area:model" \
"Architecture comparison: SegFormer" \
"Train a transformer variant under the same protocol. Two architectures
honestly compared beats five you can't explain.

Record parameter count, training cost and inference latency alongside accuracy."

# ---------------------------------------------------------------- M3
new "M3 Serving" "area:serving" \
"Tiled inference over full scenes" \
"Sliding-window prediction with overlap blending to remove seam artefacts.
Write georeferenced Cloud-Optimized GeoTIFF output that opens correctly in QGIS
with the right CRS.

Verify against a hand-checked scene before trusting it."

new "M3 Serving" "area:serving" \
"ONNX export and latency benchmark" \
"Export, verify numerical parity with the PyTorch model, then benchmark GPU vs
CPU at realistic tile sizes.

Document the result either way. If CPU is fast enough, that is a legitimate
cost-engineering finding and belongs in the write-up."

new "M3 Serving" "area:serving" \
"Modal inference endpoint" \
"FastAPI wrapped in a Modal function. Scale-to-zero, per-second billing.
Measure cold-start latency and surface it honestly in the UI rather than hiding
it behind a spinner that implies the model is slow.

Record projected monthly cost at plausible demo traffic."

# ---------------------------------------------------------------- M4
new "M4 Demo" "area:frontend" \
"Map view with pre/post swipe" \
"MapLibre GL. Curated event list, before/after backscatter swipe, model output
overlaid with adjustable threshold.

One map, one slider, one panel. Resist every additional feature."

new "M4 Demo" "area:frontend" \
"Impact analytics panel" \
"Area inundated, road length intersected, building footprints affected, from
OSM vectors. Aggregate with DuckDB spatial rather than pandas.

State uncertainty explicitly — these are model-derived estimates, not measured
counts."

new "M4 Demo" "area:frontend" \
"Live STAC inference tab" \
"User draws a bounding box, app fetches a recent Sentinel-1 scene from a STAC
API and runs inference on it. This is the part that proves it's a system rather
than a canned demo.

Rate-limit and cap area to keep hosting costs bounded."

# ---------------------------------------------------------------- M5
new "M5 Case study" "area:data,blocked" \
"Acquire and terrain-correct Nepal scenes" \
"Blocked on the M0 coverage check. Pull RTC if available, otherwise process GRD
via ASF HyP3.

Document the preprocessing chain explicitly — it's a substantial part of what
makes this defensible in interview."

new "M5 Case study" "area:model" \
"Pre-event negative control" \
"Run inference on a pre-event acquisition over the same terrain. Mask permanent
water with JRC Global Surface Water. Every remaining positive is a false
positive by construction.

Measure with and without terrain channels. This is the quantitative core of the
case study and needs no hand labelling."

new "M5 Case study" "area:docs" \
"Nepal write-up" \
"Qualitative analysis of model behaviour on the 2026 Rasuwa event. Explain the
debris-flow vs open-water mismatch, cite EMSR927 and UNOSAT 4257, and state
plainly why an IoU against those products would be meaningless.

Sober tone. Real casualties. No claim of operational or early-warning value —
frame strictly as post-event mapping."

# ---------------------------------------------------------------- M6
new "M6 Release" "area:docs" \
"Model card" \
"Intended use, training data, evaluation protocol, known failure modes,
out-of-scope uses, ethical considerations."

new "M6 Release" "area:infra" \
"Reproducibility pass" \
"Dockerfile, pinned lockfile, seeded runs, documented hardware. Fresh clone to
trained model with no undocumented steps.

Have someone else attempt it if you can."

new "M6 Release" "area:docs" \
"Demo video and v1.0.0 release" \
"Screen recording walking through the demo and the terrain ablation result.
Tag v1.0.0 with release notes."

echo "==> done"
