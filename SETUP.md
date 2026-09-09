# SETUP

One-time bootstrap. Follow in order — later steps assume earlier ones.

Delete this file once you've worked through it; it's scaffolding, not
documentation.

---

## What's in this bundle

```
CLAUDE.md                        Operating context for Claude Code. Read first.
README.md                        Public-facing project description.
SETUP.md                         This file.
LICENSE                          MIT (code only — datasets have own terms).
CITATION.cff                     Citation metadata.
pyproject.toml                   Deps, ruff, mypy, pytest config.
.pre-commit-config.yaml          Hooks: ruff, nbstripout, large-file guard.
.gitignore                       Excludes data/, checkpoints, secrets, wandb.

.github/
  workflows/ci.yml               Lint, format, types, tests. Under 10 min.
  pull_request_template.md       Forces a "why this approach" section.

scripts/
  verify_feasibility.py          M0 gate. Run this before anything else.
  seed_issues.sh                 Creates labels, 7 milestones, 24 issues.

src/waterline/                   Package skeleton (data/models/eval/inference).
tests/test_package.py            Import smoke tests so CI is green from day one.
docs/nepal-case-study.md         Scope + constraints stub for M5.
configs/                         Empty. Hydra configs land here in M1.
```

---

## 1. Fill in placeholders

Search for `<` and replace:

- `LICENSE` — `<YOUR NAME>`
- `CITATION.cff` — name fields, `<user>`
- `README.md` — `<user>` in the clone URL

---

## 2. Copy in and make the first commit

```bash
# from your empty repo directory
cp -r /path/to/waterline-starter/. .

git add .
git commit -m "chore: initial project scaffold"
git push -u origin main
```

Note the trailing `/.` — it copies dotfiles (`.github`, `.gitignore`,
`.pre-commit-config.yaml`). Without it you'll silently lose them.

---

## 3. Environment

```bash
uv venv && source .venv/bin/activate     # or: python -m venv .venv
uv pip install -e ".[dev]"               # or: pip install -e ".[dev]"
pre-commit install
```

Verify the toolchain before trusting CI:

```bash
ruff check . && mypy && pytest
```

All three should pass on a clean checkout. If they don't, fix that now — a red
CI on commit two is a bad habit to start with.

---

## 4. Branch protection

Settings → Branches → add a rule for `main`:

- Require a pull request before merging
- Require status checks to pass → select `quality` (appears after CI runs once)
- Do not allow bypassing

Solo repos don't need this functionally. You want it because it produces a
reviewable PR history, which is what a hiring manager scrolls through.

Also set Settings → General → Pull Requests → **allow squash merging only**.

---

## 5. Seed the issue tracker

```bash
gh auth login
bash scripts/seed_issues.sh
```

Creates 8 labels, 7 milestones (M0–M6), and 24 issues with acceptance criteria.
Re-running duplicates issues — edit the script rather than running it twice.

Check the board, then paste the M0 issue numbers into `CLAUDE.md` where the
milestone list is.

---

## 6. Run the feasibility gate

This decides whether the Nepal case study survives.

```bash
python scripts/verify_feasibility.py
```

It queries Planetary Computer and AWS Earth Search for Sentinel-1 over the
Rasuwa–Trishuli corridor, groups scenes by relative orbit and pass direction,
and flags orbits with a genuine pre/post pair.

Same orbit matters — comparing scenes from different viewing geometries in
steep terrain is mostly noise.

**Decision rule:**

| Result | Action |
|---|---|
| Pair on RTC | M5 proceeds as written |
| Pair on GRD only | M5 proceeds; budget time for ASF HyP3 terrain correction |
| No pair | Drop M5; substitute a mountainous European event from MMFlood |

Record the outcome in `docs/nepal-case-study.md` either way. A documented
negative finding is worth more than a silently dropped milestone.

Planetary Computer's RTC collection needs a free subscription key; GRD is open.

---

## 7. Stage the data

You're handling this manually, per `CLAUDE.md`. Expected layout:

```
data/
  mmflood/            MMFlood via TorchGeo, include_dem=True
  copernicus_bench/   CopernicusBenchFloodS1
  nepal/              Sentinel-1 over Bhote Koshi / Trishuli
  aux/                JRC Global Surface Water, OSM extracts
```

MMFlood downloads as 11 tar parts — check free disk before starting.

```python
from torchgeo.datasets import MMFlood

ds = MMFlood(root="./data/mmflood", split="train",
             include_dem=True, download=True)
print(len(ds), ds[0]["image"].shape)
```

`data/` is gitignored. Nothing in it should ever reach the repo.

---

## 8. Start work

```bash
claude update              # Opus 5 needs v2.1.219+, Sonnet 5 needs v2.1.197+
claude --model opusplan
```

Connect the GitHub connector so Claude reads issues directly rather than
inferring the plan.

Then, per issue:

```bash
git checkout -b feat/<issue-number>-<slug>
# plan mode → review the plan → implement → PR → squash merge
```

Start with M0. Do not touch frontend code until the three M0 spikes clear.

---

## Sanity checks

Before you consider the scaffold done:

- [ ] `ruff check`, `mypy`, `pytest` all pass locally
- [ ] CI green on a throwaway PR
- [ ] `main` protected, squash-merge only
- [ ] `git status` shows nothing under `data/`
- [ ] 24 issues visible, assigned to milestones
- [ ] `verify_feasibility.py` has been run and the result recorded
