# Nepal case study — scope and constraints

**Status: not started. Gated on the M0 Sentinel-1 coverage check.**

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

## Fallback

If the M0 check finds no same-relative-orbit pre/post pair over the corridor,
substitute a mountainous European event already present in MMFlood and record
that decision here.

## Tone

Sober. Post-event mapping only — no claim of predictive or early-warning
value. Cite humanitarian and agency sources. State limitations before results.
