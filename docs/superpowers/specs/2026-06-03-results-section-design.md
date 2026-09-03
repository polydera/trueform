# Results Section — Design Spec

Paper: `research/uncertainty-aware-mesh-csg/sajovic-2026-mesh-csg.tex`
Date: 2026-06-03
Status: design approved, pending spec review

## Goal

Fill the empty `\section{Results}` (currently a stub at line ~388). The section
must *prove* two claims the prose already commits to:

1. **§2.5 / §2.7 / §3** — the arrangement and per-domain bitvectors are built
   once; each boolean expression is then a single linear pass, so a family of
   booleans over the same operands reuses one arrangement.
2. **Title ("uncertainty-aware") + §2 + appendix** — classification is a vote
   that absorbs disagreement once geometry has crossed the exact boundary; the
   aggregation is MAP estimation.

The lead experiment is the one no competitor can produce (a posterior over a
component's classification). Competitive standing is secondary and deliberately
not framed as a raw-speed race (the prose already concedes EMBER is faster on
single ops).

## Baselines (same-iron available)

All runnable locally per the user: **Geogram** (Lévy 2025), **EMBER**, **CGAL**
(NEF + co-refinement, via OpenSCAD backends). So R5 can be a true single-machine
head-to-head rather than cited numbers.

Lévy's published `fibo` numbers (his Table 6, "Large" collection; for sanity /
cross-check, not the comparison itself):

| scene | Lévy (s) | CGAL coref (s) | manifold FP (s) |
|---|---|---|---|
| fibo_bunny_union | 749 | 1482 | 380 |
| fibo_bunny_diff  | 985 | 1450 | 311 |
| fibo_sphere_20   | 3   | 5    | 1   |
| fibo_sphere_100  | 44  | 69   | 18  |
| fibo_sphere_200  | 200 | 292  | 72  |
| fibo_sphere_500  | 2225| X (OOM crash) | 385 |

`fibo_bunny` = 200 bunnies in a golden-angle (phyllotaxis) distribution on a
sphere, union and difference (Lévy's teaser, ~30M facets). `fibo_sphere_N` = the
same ladder with N spheres.

## Existing infrastructure

`experimentation/` already has seeds: `csg_bunny_swarm.cpp` (536 lines),
`csg_scale_bench.cpp`, `csg_vs_boolean_bench.cpp`, `csg_vs_boolean_phases.cpp`,
`csg_nested_grid.cpp`. The Fibonacci ladder and phase breakdown are mostly a
matter of wiring these to a timing/output harness, not new pipelines.

## Paper-side hooks already inserted (these set up the experiments)

- **§2.4** — the original-left wedge apex: per-apex perturbations of
  Fig. `observed_wedge` cannot arise (apices are exact input vertices); only the
  shared edge axis perturbs, coherently across the fan, so the disturbance is
  *lessened, not removed*; the relation vote resolves the rest.
- **§3** — the separable float path `make_mesh_arrangements → make_domain_labels
  → split_into_domains`. The float labeller still runs exact predicates
  internally; materialisation has already happened upstream, so it reads
  constructed coordinates past the exact boundary, takes a bare arrangement with
  no provenance (cannot reach the original apex), and relies on the vote alone.

R1 is the experiment that earns both of these claims.

## Experiments

### R1 — float-wrong / exact-right, then the posterior (LEAD, unique)

The single measurement no other CSG paper produces. Two phases: **R1a** the
static float-vs-exact ablation (verified, locked below); **R1b** the perturbation
sweep / posterior built on top.

---

#### R1a — LOCKED: the static float-vs-exact ablation (verified 2026-06-03)

**Harness:** `experimentation/csg_wedge_votes.cpp` — replicates the
`make_domain_labels` (float) and `csg_graph` (exact int-inside) canonicalisation
front-halves so we own `get_point` and can read the per-edge canonical
permutations the library never exposes. Reports, per path: valid/total NM edges
and the number of distinct **merge-sets** (1 = clean delta).

**Verified methodology (do not relitigate):**
- **Merge-set is the ground truth**, not the permutation string. Rotation-canonical
  permutation ≡ merge-set, *bijectively* (counts match exactly). The library's
  `compute_majority_rep` (rotation-only, exact `std::equal`, **no** reversal
  grouping; both paths share it) is correct. Do **not** add reversal-equivalence:
  for K=4 a reverse is a genuine component *swap* with a different merge-set.
- **Axis-direction invariance**: flipping an edge's axis reverses the order *and*
  flips every `s(F)` → merges unchanged (proved + checked: edge gives SAME labels
  both directions). So aligning axes to the polyline isn't for correctness (merges
  are axis-invariant per edge) — it makes the string-vote equal the label-vote.
- **Pivot cannot change the ordering**: it only sets the half-plane split point;
  `rotate`-to-smallest-label erases it; the cyclic order is geometry-determined.
- The intersection of two closed meshes is **one polyline, one relation**;
  disagreements are isolated interior near-degenerate edges, correctly out-voted.

**Canonical scene: two unit spheres (60×60), centers 0.7 apart** → one
404-edge intersection circle.
- **EXACT int-inside: 404/404, one merge-set — a perfect delta.**
- **FLOAT: 403 : 1** — all valid, but one wedge flips.

The single flipped wedge, identified and matched by world location:
```
FLOAT  WRONG edge: order [0,3,2,1], min|sep| = 1.0e16   (vs clean wedge 1.8e22 — ~10^6x closer to coplanar)
       world location (0.350, 0.254, -0.901)
EXACT  edge nearest that spot: dist 0.0000, order = MAJORITY  (resolved correctly)
```
Same near-degenerate wedge; the 24-bit float round-trip flips its sign, exact int
constructions + original-vertex apex resolve it. The vote then absorbs the lone
float flip (403:1). **This is R1's two messages on one identifiable wedge.**

Regime (offset sweep): `(1,0,0)` both dirty (symmetric degeneracy); `(0.7,0,0)`
and `(1.5,0,0)` float-dirty / exact-clean; `(1.3,0,0)` the mirror (exact catches
one, float clean); any off-axis nudge → both clean. So it's geometry-specific
which path hits a degeneracy — float is not universally noisier — but `(0.7,0,0)`
is the cleanest "float wrong, exact right" example (nothing dropped).

---

#### R1b — Perturbation sweep / posterior (next)

**Mechanism.** The arrangement topology is built combinatorially with indirect
predicates and never sees coordinates; constructed points enter only when
classification votes. So perturbing the materialised points *after the graph is
built, before classification* injects exactly "a non-exact construction that
crossed the exact boundary" while the topology stays fixed.

**What we perturb: the non-manifold edges we operate on** — i.e. the
materialised intersection points that serve as NM-edge endpoints (the
constructions), *not* the wedge apex. The apex is an exact original vertex and
is protected by design (§2.4); perturbing it would be perturbing something exact
by construction. Each materialised point is jittered **once per draw** (it has a
single position), so every NM edge that shares it sees the same shifted
coordinate — coherent, not independent per-edge noise.

- Integrated path: wrap the constructed-point materialiser in `csg_graph.hpp`
  (the `get_point` lambda, `csg_graph.hpp:105`; perturb only the
  `vertex_source::created` branch — these are the NM-edge endpoints) with a
  uniform box jitter of half-width ε in grid units. Topology is frozen above
  (built at lines 100–103); only the descriptor / volumes / seeding consume the
  perturbed points. The apex is fetched as an original vertex, so it is never
  touched.
- Float path: run the same scene through `make_domain_labels`. Perturb the
  *same physical points* (the intersection points). With no provenance, its apex
  is the loop's own off-edge vertex, which may itself be one of those perturbed
  points — so axis and apex are both exposed. This is the ablation: same
  perturbation, different exposure.

**Ablation (two curves, two messages).** Run the identical perturbation through:
- `csg_graph` (apex-protected — only the shared axis is non-exact), and
- `make_domain_labels` (float — apex exposed, no provenance).

The gap between the two curves measures the value of the original-left-apex
design (§2.4). Both holding correct well past where a single deterministic
`orient3d` would flip measures the vote (§2 + appendix).

**Scene.** Minimal: one component whose label rests on a small, controllable
wedge load (candidate: single box ∩ single tetra, or two boxes in coplanar
kiss). Ground-truth label known analytically.

**What we record (two views that should coincide).**
1. *Analytic posterior.* Per-component wedge-vote tally (N_in, N_out) →
   Beta(N_in+α, N_out+α). Plot density widening with ε, MAP staying correct,
   credible interval + breakpoint.
2. *Monte-Carlo posterior-predictive.* Over M perturbation draws at each ε, the
   empirical label / vote distribution.

Their agreement is itself a result: the perturbation statistics behave like the
Bernoulli wedge draws the model assumes, empirically validating the
Beta–Bernoulli aggregation rather than only using it.

**New code required.** Expose per-component vote tallies (N_in, N_out) out of the
classifier (`compute_majority_rep` / `seed_inclusion_bits` for the integrated
path; the `make_domain_labels` vote for the float path). Currently only the hard
MAP label survives. This is the one non-trivial instrumentation task.

### R2 — Aggregation robustness at population scale

Same perturbation mechanism, aggregated over *all* components of a real scene
with analytic ground truth (nested / fibo spheres). Corrupt fraction p of wedge
votes; plot fraction of components correctly classified vs p. Stays ~100% until
p → 0.5 (majority breakdown); overlay a deterministic single-test baseline that
flips at the first bad vote. R1 is the microscope, R2 the population view.

### R3 — Build-once / extract-many (architecture headline)

One `make_csg_graph` build; then the marginal cost of each additional
`make_csg_mesh` expression (∪, −, ∩, k-ary, sub-expressions). Plot total time vs
number-of-expressions: trueform flat after the build, per-expression baselines
linear. Anchor against Lévy running union and difference as two separate full
builds (749 s + 985 s on fibo_bunny) where we build once.

### R5 — Fibonacci ladder, same-iron competitive standing

Re-run trueform vs Geogram / EMBER / CGAL-NEF / CGAL-coref on
`fibo_sphere_{20,100,200,500}` and `fibo_bunny`. Report wall-clock **and peak
memory**. Highlight: we survive 500 where CGAL OOMs; concede EMBER's raw speed,
pair it with our amortization (R3) and extra outputs. Seeds:
`csg_bunny_swarm` + `csg_scale_bench`.

### Supporting (include as space allows)

- **R4 — N-ary vs pairwise chaining.** Single global arrangement vs the same
  model as a pairwise CSG tree; linear vs super-linear. (`csg_vs_boolean_bench`.)
- **R6 — Degeneracy correctness.** Coplanar stacks, nested-no-contact,
  inconsistent winding: correct where CGAL "manifold" FP / NEF fail.
- **R7 — Phase breakdown** (extraction is a sliver; Lévy Table 3 analogue,
  `csg_vs_boolean_phases`), **thread scaling** (Stage-2 embarrassingly
  parallel), **free secondary outputs** (cross-section curves, per-domain
  volumes that Lévy gets only as a co-refinement side-effect).

## Section spine (what actually ships)

**R1 + R2 + R3 + R5** as the committed spine. R4/R6 as a half-page each, R7 as a
figure or two, included if space allows.

Ordering in the paper:
1. R1 (lead with the unique measurement; figure = posterior fan, two pipelines).
2. R3 (the architecture payoff the prose promised).
3. R5 (competitive standing, table + memory).
4. R2 (robustness at scale, backs the title).
5. R4 / R6 / R7 as space allows.

## Open decisions

- **R1 scene:** SETTLED for R1a — two unit spheres (60×60), centers 0.7 apart
  (`experimentation/csg_wedge_votes.cpp`): float 403:1, exact 404:0, one
  identifiable flipped wedge. The perturbation phase (R1b) may want an even
  smaller hand-built scene for a legible single-component posterior; revisit when
  the tally instrumentation lands.
- **Perturbation unit:** box jitter on the materialised NM-edge endpoints
  (intersection points), once per point per draw. Integer-grid units for the
  integrated path, float for `make_domain_labels`; keep magnitudes comparable
  across the ablation. Apices are never perturbed (exact originals).
- **α (Dirichlet prior):** symmetric, value for the plotted posteriors. MAP is
  α-independent (appendix); α only affects the credible-interval width shown.

## Out of scope

- Re-deriving competitor pipelines; we run their published binaries.
- Snap-rounding / output robustness studies (separate topic).
- Any change to the classification algorithm itself — R1 only *observes* it (plus
  the read-only tally export).
