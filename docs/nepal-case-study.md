# Nepal case study — scope and constraints

**Status: GO. M0 coverage check passed on 2026-09-10 — see below.**

## The event

On 26 August 2026 a glacier collapse in the Langtang range near the
Nepal–China border triggered a debris flow that travelled roughly 100 km down
the Lende Khola, Bhote Koshi and Trishuli rivers, devastating Rasuwa, Nuwakot
and downstream districts. Hundreds died.

## Why this is a qualitative case study and not a benchmark

Three reasons, all of which need stating plainly in the final write-up:

1. **Target mismatch.** The model segments smooth open water, which returns
   low radar backscatter. A debris flow of mud, rock and ice is rough and
   returns bright. The model is not expected to detect it.

2. **Reference mismatch.** The published delineations — Copernicus EMS
   activation EMSR927 and UNOSAT product 4257 — map *mudflow and rockflow
   extent* from optical imagery, not water extent. An IoU computed against
   them would measure a category mismatch, not model quality.
   **Do not compute or report that number.**

3. **Terrain.** Steep Himalayan topography produces radar shadow, which is
   dark and therefore reads as water to an untreated model. This is the
   failure mode the M2 terrain channels target.

## The quantitative component

The pre-event acquisition is a free negative control. With no flood present,
every positive prediction over the same terrain is a false positive by
construction, once permanent water is masked using JRC Global Surface Water.

Measure that false-positive rate with and without terrain channels. This
requires no hand labelling and tests the M2 hypothesis directly.

## M0 feasibility outcome

Run: `scripts/verify_feasibility.py`, 2026-09-10.

**Result: best case on the decision rule.** Three same-relative-orbit pre/post
pairs exist, and they're available on the **RTC** collection (terrain
correction already applied by the provider), not just raw GRD. The decision
rule was: pairs on RTC → M5 proceeds as scoped; pairs only on GRD → proceed,
but budget time for manual terrain correction via ASF HyP3; no pairs → drop
M5. This is the first case, so M5 proceeds as originally scoped — no ASF
HyP3 detour needed.

Checked three ways (Planetary Computer GRD, Planetary Computer RTC, AWS Earth
Search GRD) — all three returned the same 19 underlying Sentinel-1 scenes over
the AOI, confirming the coverage isn't an artefact of one provider's catalogue.

| Orbit | Direction | Pre-event gap | Post-event gap | Post-event acquisitions |
|---|---|---|---|---|
| 19  | descending | 1 day before  | 10 days after | 1 (2026-09-05) |
| 121 | descending | 6 days before | 5 days after  | 1 (2026-08-31) |
| 85  | ascending  | 9 days before | 2 days after  | 2 (2026-08-28, 2026-09-09) |

`torchgeo` 0.10.0 confirmed working; `MMFlood` and `CopernicusBenchFloodS1`
both reachable.

**Not yet decided: which orbit to use as the primary pair.** Orbit 19 gives
the tightest pre-event baseline (1 day), orbit 85 the tightest post-event
capture (2 days) plus a second post-event scene for a rough temporal check.
Picking one affects what the case study actually shows, so this is deferred
to M5 planning rather than decided here.

## Fallback (not needed)

If the M0 check had found no same-relative-orbit pre/post pair, the plan was
to substitute a mountainous European event already present in MMFlood. Kept
here for the record — the coverage check passed, so this wasn't invoked.

## Tone

Sober. Post-event mapping only — no claim of predictive or early-warning
value. Cite humanitarian and agency sources. State limitations before results.
