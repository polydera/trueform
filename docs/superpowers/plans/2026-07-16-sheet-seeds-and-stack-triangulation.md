# Coincident coplanar walls: two fixes

> Working plan, 2026-07-16. Branch: `fix/coplanar-walls` off `feature/public`.
> Order of work: Fix 1 (sheet seeds + universe) first, Fix 2 (mesh-path
> stack triangulation) after. Probe for everything:
> `experimentation/probe_opposing_sheets.cpp` (box + plane + copies, both
> paths, TF_DUMP_MEMBERSHIP=1 for the membership dump).

## The scene that exposes both

Box [-1,1]^3 + z=0 plane sheet + a coincident copy of the plane (same or
reversed winding). Ground truth: two 4.0 halves; outside excluded under
default flags; under `domain_config::none`, four raw cells with faithful
per-operand sheet bits.

## Established facts (all verified in probes, 2026-07-16)

- Arrangement is IDENTICAL in every scenario: 8 NM edges, 1 curve, all
  fans K=4. Coplanar dedup is correct for both windings.
- Raw partition is correct in every scenario.
- Graph path: coplanar duplicate loops are DROPPED at build
  (`arrangement_graph::loop_labels` = none_label for dead loops);
  `coplanar_pairs()` = (survivor, dead, reversed_flag); emission
  re-creates dead copies from the survivor's triangulation, flipping
  winding by the flag. True for cdt AND refined_cdt (dedup is in the
  arrangement, not the triangulator).
- Mesh path (`make_mesh_arrangements` / `make_polygon_arrangements` →
  `triangulate_arrangement_cuts`): NO arrangement_graph, NO coplanar
  pairing; every loop triangulates independently. Aligned stacks match
  only because identical winding gives identical CDT; reversed stacks
  get mirrored CDTs → mismatched diagonals → the wall never seals →
  observed: 8.0-volume non-manifold "uncut" cell + 4-tri zero-volume
  pillow.
- Sheet bits are seeded by `winding_side(tagged_forms[t], ...)` — the
  operand's ORIGINAL faces. Faithful per-operand semantics is the
  intended and implemented behavior.
- The reversed-pair graph-path flakiness (10-run test: 8x correct 2
  cells, 2x leaked -8 universe) is a DEGENERATE EVALUATION: the outside
  bundles' seed points lie exactly ON the coincident duplicate sheet's
  surface; winding_side has no defined side there; resolution rides on
  bundle/seed enumeration order.
  - Outcome A (bits {0,1,6,7}): degenerate answer lands both sheet bits
    on one side; zero row exists; exclusion works by luck.
  - Outcome B (bits {2,3,4,5}): faithful answer; opposite sheets tile
    space; NO zero row; `bits == 0` universe test finds nothing; the
    outside leaks as the inverted -8 cell.

---

## Fix 1 — deterministic faithful sheet seeding + fuse-clears-bits

### 1a. On-sheet seed classification (the determinism fix)

Where: `include/trueform/csg/graph/seed_inclusion_bits.hpp` — the sheet
batch block (~L315-345) calling `winding_side`.

Design: when a bundle's seed lies on a wall that is coplanar-shared with
sheet `t`, do not ask winding. Decide topologically:

    behind_t(bundle) = (bundle side vs survivor wall normal)
                       XOR (operand t opposes survivor)

Inputs that already exist:
- `arrangement_graph::coplanar_pairs()` gives (survivor, dead, reversed)
  per folded loop; descriptors give each loop's (tag, object).
- The bundle's side of its own wall is what the seed anchors anyway
  (domain_of_side).
- Detection of "seed's wall is shared with sheet t": the survivor loop's
  coplanar-pair run lists the dead loops and their tags; if any dead
  (or the survivor itself) has tag t, the wall is shared with t.

Sketch:
1. Build per-loop shared-tag info from coplanar_pairs: for each
   surviving loop, the set of (tag, reversed) folded into it (plus its
   own tag, reversed=0).
2. In the sheet-batch pass, for a (bundle, sheet t) pair whose seed
   face's loop shares tag t: skip winding_side; set
   parity = side XOR reversed(t vs survivor). The side bit convention
   must match what the side-anchoring writes below (verify against the
   existing code that writes parity into inc bits per side).
3. All other (bundle, sheet) pairs unchanged.

Open questions to resolve while implementing:
- Seed-to-loop mapping: the seed is placed on a face of the bundle;
  need the loop id of that face at the point where sheet_pairs are
  collected (the cast callback already knows cut_face_bundle — check
  what identifies the seed's carrier face).
- The pair (bundle, t) where t's geometry is coincident with SOME wall
  of the bundle but the seed sits on a DIFFERENT face of the bundle:
  winding_side is then non-degenerate and fine. Only redirect when the
  seed's own carrier is the shared wall. Alternative, more robust rule:
  if bundle's component contains ANY wall shared with sheet t, classify
  topologically (the bundle is entirely on one side of t by planarity
  of the shared wall — NOT true in general: a bundle can extend beyond
  the shared plane). DECISION: redirect only when the seed's carrier
  face is on the shared wall; if the seed can be chosen elsewhere
  (bundle has non-shared faces), an even simpler fix is seed SELECTION:
  prefer a seed on a face NOT coplanar-shared with any sheet. Evaluate
  both; seed-selection-first is less invasive but fails when ALL faces
  of the bundle are shared (sheet-only bundles — exactly our probe's
  flap bundle). So: seed selection as fast path + topological
  classification as the complete answer.

### 1b. Open-fuse clears the fused sheet's bits (the universe fix)

Where: `include/trueform/csg/graph/compute_domain_membership.hpp`.

Design: when `ignore_open_fragments` unites the two sides of open
component `c` with tag `t`, record bit `t` to be CLEARED in the coarse
representative of the united class. After the union-find and dense
numbering:

    rep_bits(k) = OR of constituent bits  (replaces last-nonzero-wins)
                  AND NOT cleared_bits(k)
    is_outer(k) = (rep_bits(k) == 0)      (replaces zero-constituent scan)

Rationale: a fused sheet no longer partitions, so "behind t" is
undefined for the fused domain. Single sheet: {0}|{s1} minus s1 = 0 —
identical to today. Opposite pair: {s1}|{s2} minus both = 0 — leak
gone, deterministically, regardless of 1a's outcome. ignore OFF: both
outside halves keep faithful bits and are KEPT — documented semantics,
not a leak.

Need the tag of an open component: descriptors know each loop's tag;
components→tag via `compute_bundle_tag_index` outputs already on the
descriptor (`tag_of_component`). Verify availability in the membership
function's inputs; if absent, pass it in (it is computed during build).

Behavior preserved: volume-only graphs untouched (nothing clears);
structural (`with_self`) branch untouched; single-sheet docs paragraph
behavior identical.

### Consequences to re-verify after 1a+1b

- Expressions: `op(sheet1)` vs `op(sheet2)` on the reversed pair select
  OPPOSITE halves (today they nondeterministically select the same).
- `exclude_outer_shell` semantics table in docs stays true.
- The `rep_bits` change from last-nonzero-wins to OR-minus-cleared can
  change which bits a fused MIXED domain reports in imap.inclusion —
  audit test expectations (tests/csg/test_csg_domains.cpp sheet cases,
  test_csg_sheets.cpp).

### Tests for Fix 1

- New suite cases (tests/csg/test_csg_sheets.cpp or test_csg_domains):
  box + sheet + reversed coincident sheet: default = 2 cells; none = 4
  cells with faithful bits ([s1-behind] vs [s2-behind] on opposite
  sides); op(1)/op(2) select opposite halves; run-stability implicitly
  pinned by determinism of the classification.
  Same-orientation duplicate: identical expectations with both bits on
  one side.
- Probe: 20x rerun loop of probe_opposing_sheets must give identical
  output every run (add a quick shell check, not a suite test).
- Full suites: csg, cut, topology, intersect, clean. Sheets fuzz
  (test_expression_fuzz) green.
- Geological matrix probe (probe_geo_domains): 89/89/91/91/89/91
  unchanged.
- Perf: bench_stages_csg corpus A/B — seeding change adds a per-loop
  shared-tag lookup; expect noise-level. Gate per HOW_WE_WORK.

---

## Fix 2 — mesh-path stack triangulation sharing (MAPPED 2026-07-16)

Problem: the mesh path triangulates every loop independently;
opposite-winding coincident stacks get mirrored CDTs (Delaunay ties —
cocircular quads — break by insertion order), the duplicate faces stop
being vertex-identical, and downstream dedup (clean) and stack
resolution (make_domain_labels keep-one) never fire: the wall does not
seal.

### The map — investigation results (2026-07-16, second pass)

CENSUS: who holds/uses cut_graph today.
- Workers: polygon_arrangements_worker (self) and
  mesh_arrangements_pair_worker get cg UNCONDITIONALLY from
  build_self_pipeline / build_exact_pipeline; mesh_arrangements_n
  builds a LOCAL cg only inside its want_curves branch.
- cut_graph products and their consumers:
  - connectivity_per_face_edge: boolean pipeline only
    (make_classifications, classify_missing, partition/make_labels).
  - coplanar_pairs + coplanar_mask: boolean classification only.
  - intersection_edges / is_intersection_edge: boolean labels + every
    want_curves branch.
- KEY FINDING: the plain (no-curves) ARRANGEMENT outputs use NO
  cut_graph product at all. The pair/self workers' unconditional
  cg.build is inherited waste — the shared pipeline builders serve the
  boolean path, which does need cg. The N worker already avoids it on
  the main path. cut_graph::build = flat ids + face_membership +
  face_link_per_edge + coplanar pairs + intersection-edge set — real
  cost, entirely unused by a plain arrangement call.

OPTIONS for the fold source:
- (A) full cut_graph in the N worker too: adds the whole build to the
  heaviest path for one product; against the N worker's existing
  lazy design.
- (B) staged cut_graph (connectivity+coplanar now, edges on demand):
  still drags connectivity in only as the candidate filter.
- (C) STANDALONE coplanarity detector, no cut_graph (Žiga's hunch —
  confirmed viable): tf::cut::detect_coplanar_loop_folds(fc, mode)
  -> (dead, survivor, reversed) folds. NO HASHES (house rule:
  everything is a sort): per loop, key = its tag-wrapped identity
  sequence SORTED (multiset order, small per-loop sort into a flat
  buffer with offsets) — the same set-canonical key pattern as the NM
  relation grouping; one argsort of loop ids by (size, key lex), group
  equal runs, exact compare_faces within runs decides true coincidence
  and the orientation flag, min-loop-id as survivor. Needs ONLY fc.loops() + descriptors — no
  flat ids, no face_membership, no connectivity. O(sum |loop|) key
  materialisation + one argsort; runs of equal keys are the stacks.
  Mode parity with cut_graph: between skips same-tag, self skips
  same-object; tag-wrapped keys identical (created -> -1, original ->
  tag); orientation = compare_faces sign.

DECISION LEAN: (C). One detector, three workers consume it directly.
Queued follow-up cleanups (separate commits, not this fix):
  1. cut_graph::build_coplanar_pairs delegates to the standalone
     detector (drops its connectivity dependence for detection).
  2. Plain-arrangement calls stop building cg at all: pipeline
     builders return it lazily / workers build it in the curves
     branch only — straight perf win for arrangements, booleans
     unaffected.
  3. arrangement_graph's private detector delegates too (the ag
     unification, biggest blast radius, last).
Perf gate: detector cost A/B on corpus + geological (it runs on every
arrangement build; expect ~linear hashing, verify).

### The emit — re-triangulate the survivor's loop, no output copying

Key simplification over the graph path's copy-from-survivor: in the
triangulator's per-loop lambda, a DEAD loop
  1. builds its points from the SURVIVOR's loop (same vertex order),
  2. projects with the SURVIVOR's projector (own-face projection_axes
     could pick different axes -> different 2D -> different CDT; the
     survivor's axes are the canonical ones),
  3. triangulates — the CDT is deterministic, so this reproduces the
     survivor's triangulation exactly, without sharing outputs across
     parallel items,
  4. maps vertices through its OWN desc.tag (identities equal, mapping
     valid), emits its OWN (tag, object) labels,
  5. flips each triangle's winding when `reversed`.
No two-phase, no offsets, generic_generate stays embarrassingly
parallel. Cost: one redundant CDT per dead loop — stacks are rare.

### Where it lands

triangulate_arrangement_cuts + triangulate_partition_cuts gain an
optional per-loop fold lookup (defaulted empty -> zero change for
existing callers). Consumers (7 call sites): make_polygon_arrangements,
make_mesh_arrangements (pair + N), make_boolean, make_boolean_pair,
embedded_intersection_curves, embedded_isocurves, make_isobands.
Scope: pass folds in the two ARRANGEMENTS paths first (the domain
workflows); booleans and embedded paths after the arrangements
validate — same parameter, separate commits.

### Thing two: free-path stack resolution on reversed twins

Once twins are vertex-identical, verify and if needed extend:
- cleaned() unique-faces mask: doc says either-winding already — pin
  with a reversed-stack test.
- make_domain_labels stack keep-one (compute_face_stacks +
  resolve_face_stacks): pairing keys on face identity (either winding
  per are_faces_equal), but the orientation lift / survivor
  materialization was only proven on same-winding stacks. Reversed
  stack: which winding survives, and does orient_faces_consistently +
  the flip-tracked label lift stay correct. Extend
  tests/topology/test_domain_labels.cpp.

### Tests / gates

- probe_opposing_sheets free arms: reversed copy -> 2 domains 4+4 in
  BOTH clean modes (today: 8.0 non-manifold + 0-volume pillow);
  3-stack mixed windings free arm added.
- Suite: reversed-stack cases for make_polygon_arrangements and
  make_mesh_arrangements (watertight halves after domain labels).
- Free==graph parity on the probe scenarios.
- Geological matrix unchanged (89/91 = number-systems residual, NOT
  this fix's target).
- Full suites + perf A/B: detection cost measured on the corpus
  (bench_stages_csg + boolean bench); emit is unchanged for fold-free
  input so expect noise; gate per HOW_WE_WORK.

---

## Protocol (per HOW_WE_WORK)

- Branch `fix/coplanar-walls`; lib changes gated by full suite runs
  (fresh C++17 build) + probes + geological matrix + perf A/B before
  any commit; tests land WITH the fix commit; experimentation stays
  untracked; MSVC lambda checklist on every new lambda.
- Fix 1 first. Fix 2 only after Fix 1 is committed and green.

---

## STATUS 2026-07-16 evening

FIX 1 LANDED on fix/coplanar-walls:
- 1b = 0ee20d33 (fuse clears fused sheet bits; rep = OR-minus-cleared;
  is_outer = rep==0; needed the coplanar fold-ins too — clearing only
  the survivor's tag was insufficient, the dead duplicate has no
  component to fuse).
- 1a = 4060d8a2 (anchor_sheet_sides gains (component, tag, reversed)
  fold entries from coplanar_pairs; folded sheets anchor through the
  carrying wall, mirrored by the reversed flag). The REAL
  nondeterminism source was not seed choice: winding_side accumulates
  solid angles in double via thread-local partials, and an on-sheet
  query sits exactly at the pi threshold (the coincident wall
  contributes +-pi noise terms) — reduction order decided the sign.
  The fold anchor overwrites that polarity; degenerate winding becomes
  irrelevant wherever a fold exists.
- Gates: csg 84 cases / cut / topology / intersect green; geological
  matrix unchanged (89/89/91/91/89/91); probe 10/10 deterministic
  default extraction AND faithful raw bits; op(1)/op(2) select
  opposite halves 10/10.
- Residual (documented, not fixed): winding_side's float reduction is
  still order-dependent in general; clean queries have huge margins,
  and fold-covered degenerate queries are now overridden. A seed
  landing exactly on an UNRELATED sheet's surface (no fold) would
  still be a coin flip — no known repro; revisit if one appears.

NEXT: Fix 2 (mesh-path stack triangulation sharing) per plan above.

## STATUS 2026-07-16 late

FIX 2 LANDED: 71c96d96 (detector: min-edge key (vmin,vmax,size), edge
leads the comparison, anchored no-search verification; parity with
cut_graph on all scenes, 0.8ms/52k loops) + 73e732df (fold-aware
triangulate_*_cuts + wiring, scope::all). Suites green (cut 143,
topology 264, csg 86); geo + sphere matrices unchanged; keep-dups
reversed stack = 2 closed 4.0 halves, 5/5 stable.

REMAINING:
- dedup arm of the reversed scene = MINIMAL REPRO of the pre-existing
  make_domain_labels TBB race (2/10 MT wrong, ST 10/10 correct,
  deterministic input). Separate project (project_domain_labels_race)
  -- now with the best repro it has ever had.
- cut_graph::build_coplanar_pairs delegation to the detector (cleanup
  1) + plain-arrangement cg elision (cleanup 2) + ag unification
  (cleanup 3) -- queued.
- Perf A/B on corpus (bench_stages / boolean bench) before merge.
- Booleans + embedded paths fold wiring (second wave).
