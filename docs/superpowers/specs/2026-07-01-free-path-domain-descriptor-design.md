# Free-path domain descriptor — design

## Problem

The library has two ways to reason about the volumetric domains an
arrangement of N forms carves out of space:

1. **CSG path.** `tf::make_csg_graph(forms)` builds an implicit
   `csg_graph`, which internally computes a **per-domain operand-inclusion
   lattice** (a bitvector per domain, bit *i* = "inside form *i*"). Any
   boolean `tf::csg::expr` selects domains against those bits; that is the
   whole selection story (`make_csg_domains`, `make_csg_mesh`).

2. **Free / arrangement path.** `tf::make_mesh_arrangements(forms)` →
   merged mesh + `tag_labels` (form per face); `tf::make_domain_labels(mesh)`
   → domains (per-face per-side domain id, `n_domains`, `outer_shell`);
   `tf::split_into_domains` → one mesh per domain. This pipeline is
   *pluggable* — a user can enter at any stage — **but it has no operand
   classification.** There is no way to ask "domains inside form X" or
   "inside X and not Y" the way a CSG expression can.

`tf::make_sidedness_relations(mesh, tag_labels)` gives only a *local*
per-component classification: for each manifold-edge component, which side
it is on **against operands it shares an intersection edge with**. It says
nothing about forms a component does not touch, or about deep nesting. It
is the *seed* of a classification, not the classification.

**The missing piece:** a free-path structure that assigns each domain its
full operand-inclusion bits — the free-path twin of the CSG lattice — so a
user can do boolean *and* manual/graph selection off the arrangement
without building a `csg_graph`.

## The two existing classification paths (analysis)

The free and CSG raycasts are **structural twins**.

**Free path** — `tf::topology::domains::make_nesting_merges(polygons,
fragment_labels, bundle_labels, domain_of_side, domain_volumes, get_point,
merges, removed_domain)`:
- Per bundle, `outer_env` = the most-negative-volume incident domain.
- **Single-bundle → returns before any cast** (line 118). No nesting is
  possible with one bundle.
- Multi-bundle: bbox-containment candidate pairs, then one segment cast
  (bundle seed → far point) through the merged mesh; each hit face's two
  adjacent domains get a `hit{bundle, domain, dist_sq}`. Per
  `(bundle, domain)`, odd hit-parity ⇒ the domain encloses the seed; the
  closest such is the seed's physical domain. Emits nesting + universe
  merges. Returns `root_anchor`.
- It performs the **nesting reduction only**.

**CSG path** — `tf::csg::graph::seed_inclusion_bits(...)` →
`tf::cut::propagate_inclusion_bits(...)` → `tf::cut::anchor_sheet_sides(...)`:
- `seed_inclusion_bits` is the *same* raycast (same `hit_t`, same
  `outer_env`, same single-bundle early-out at line 142) but reduced **two
  ways in one walk**: nesting hits **and** per-form crossing parity →
  the outer-env domain's **absolute inclusion bits**. It returns the set
  of anchored domains as BFS roots. Sheets are side-classified with
  `winding_side` (not the ray). `null_seed` = globally most-negative
  volume, anchored at all-zero.
- `propagate_inclusion_bits` XOR-BFS's the bits across `domain_of_side`:
  crossing component `c` XORs by `B(c)` (its form bit + any coplanar
  fold), spreading every seed to every domain.
- `anchor_sheet_sides` renormalizes sheet bits so the set bit lands on
  the fragment's side-1 (stored-winding-interior = behind the normal)
  domain, per connected region of the domain graph.

So the free path already does ~90% of the work. The delta to a full
free-path classification is exactly:
- **(a)** feed `tag_labels` into the cast so it *also* bins per-form
  parity — a second reduction in the *same* walk;
- **(b)** the XOR-BFS propagate;
- **(c)** the sheet anchor.

## Design

### The type — `tf::domain_descriptor<Index>` (in `cut/`)

```cpp
template <typename Index> struct domain_descriptor {
  tf::domain_labels<Index>   labels;      // per-face per-side domain id, n_domains, outer_shell
  tf::cut::domain_inclusions inclusion;   // per-domain bitvector, words = ceil(n_tags/32)
  Index                      n_tags = 0;  // number of input forms
};
```

`inclusion.make_range()[d]` yields domain `d`'s bits; bit `i` = "inside
form `i`" (volume: winding-parity solid; sheet: behind the normal — same
semantics as CSG). `labels` gives the domain adjacency for free (each face
connects its two side-domains), so a user does boolean selection (over the
bits) *and* manual/graph selection (over the labels) from one struct.

Lives in `cut/` (sibling of `arrangement_descriptor`, reuses the
`cut/arrangements/` inclusion machinery, consumes `tag_labels` from
`cut::make_mesh_arrangements`).

### The builder — `tf::make_domain_descriptor(polygons, tag_labels, sheets, config)` (in `cut/`)

- `polygons` — the arrangement surface (oriented, non-manifold).
- `tag_labels` — range, per face → form tag `[0, n_tags)`
  (`make_mesh_arrangements` returns it).
- `sheets` — range of form tags to treat as sheets (empty = all volumes),
  the free-path `is_sheet`.
- `config` — `tf::domain_config`, **exactly as `make_domain_labels`**.

It is a **superset of `make_domain_labels`**: same partition, plus
inclusion. Semantics match CSG: a sheet-tagged open fragment does **not**
self-merge its two sides (even in Mode 2); everything else obeys `config`.

### No double-compute — shared core (the critical requirement)

`make_domain_labels` already computes the partition, bundles, **volumes**,
and the **nesting raycast**. That is exactly what the descriptor also
needs. Factor it so nothing is computed twice, and `make_domain_labels`
stays lean (never pays for tags/inclusion).

1. **`resolve_domain_partition(polygons, config)` — new shared core**
   (`topology/domains/`). Extracts everything `make_domain_labels`
   currently does *up to but not including* `lift_to_domain_labels`:
   `fragment_labels`, `domain_of_side`, `n_domains`, `bundle_labels`,
   `domain_volumes`, nesting result, `removed_domain`, `root_anchor`.
   Returns a small intermediate struct. Single source of truth.

2. **`make_domain_labels` = `resolve_domain_partition` →
   `lift_to_domain_labels`.** Same work, same cost, refactored onto the
   core. Computes nothing extra.

3. **The raycast is the shared seed, extended like `seed_inclusion_bits`.**
   Generalize `make_nesting_merges` with a `bool WantInclusion` template.
   With `WantInclusion=false` (the labels path) it behaves exactly as
   today (`if constexpr` guards the extra work → zero added cost). With
   `WantInclusion=true` it also takes `tag_labels`, `n_tags`, `sheets`,
   and — in the *same* cast that produces the nesting hits — bins per-form
   crossing parity into the outer-env domains' inclusion bits, and returns
   the anchored seed domains (BFS roots). **Single cast, two reductions;
   single-bundle skip preserved** (records `null_seed`, no ray).

4. **`make_domain_descriptor` = `resolve_domain_partition`
   (seed-with-tags) → `propagate` → `anchor` → `{lift'd labels,
   inclusion}`.** It adds *only* propagate + anchor over shared results —
   the exact incremental work the CSG steps add over the
   descriptor+volumes they already have.

Net: **one partition, one raycast, one volumes pass**, shared by both
functions; the descriptor does the two extra steps; nothing is
duplicated.

### Generalize propagate + anchor to run off `domain_of_side`

`propagate_inclusion_bits` today reads `desc` (`domain_of_side`,
`n_domains`, `tag_of_component`) **and** `ag`/`fc` — but the `ag`/`fc`
touches are only to build `B(c)` (per-component form-bit mask + coplanar
folds). Add a `cut` overload:

```cpp
propagate_inclusion_bits(domain_inclusions &inc,
                         const buffer<Index> &domain_of_side,
                         Index n_components, Index n_domains,
                         const buffer<uint32_t> &B,   // words_per_domain per component
                         const buffer<Index> &seeds);
```

The free path supplies `B(c)` directly: for fragment `c`, the single bit
of its faces' `tag_labels`. The CSG caller keeps its current overload
(builds `B` from `ag`/`fc`, then delegates to this core).

`anchor_sheet_sides` already reads only `desc.domain_of_side`,
`desc.tag_of_component`, `desc.bundle_of_component`, and `is_sheet` — no
`ag`/`fc`. Generalize it to take those four buffers directly so the free
path can call the identical logic.

### Sheets

- **Partition:** a sheet fragment's two sides never self-merge (mirror
  CSG's `is_sheet` in the descriptor merge step). Volume open fins obey
  `config` as today.
- **Seed:** volume forms via the ray parity (as above); sheet forms via
  the same crossing parity through the arrangement's sheet faces (bin by
  `tag_labels`), giving a consistent per-region value.
- **Anchor:** `anchor_sheet_sides` fixes the absolute orientation (set bit
  on the behind-the-normal side).

### Out of scope: selection

`make_domain_descriptor` returns the descriptor and stops. **No
expression evaluator, no keep-mask, no split baked in.** The user reads
`inclusion.bits` per domain and applies their own boolean/manual logic. A
`select`/`split_by(descriptor, predicate)` helper (possibly reusing the
`csg::expr` evaluator) is a clean, separate follow-up — not this project.

## Known wrinkle

Coplanar-shared faces: `make_domain_labels`' coplanar dedup can leave one
survivor face carrying one form's `tag_labels` where two forms coincide.
The free-path `B(c)` then misses the second form's bit at that overlap —
the same limitation `return_source_ids` has. Exact for
non-coplanar-sharing inputs; a per-face tag *set* is a future extension.
CSG handles this via `ag.coplanar_pairs`; documenting the parity, not
fixing it here.

## Testing

- **Parity vs CSG.** For a set of forms, `make_domain_descriptor`'s
  per-domain inclusion bits must match `make_csg_graph`'s `inclusion()`
  for the corresponding domains (map by representative interior point or
  by signed volume). Covers volume-only and with-sheets scenes.
- **`make_domain_labels` unchanged.** The refactor onto
  `resolve_domain_partition` must leave every existing `make_domain_labels`
  test green (partition + `n_domains` + volumes identical).
- **Selection sanity.** Given the descriptor, a hand-written "inside form
  X" filter over the bits + `split_into_domains` reproduces
  `make_csg_domains(graph, tf::csg::op(X))` cell count/volumes.
- **Single-bundle skip.** A single closed form (one bundle) produces
  correct bits without a raycast (the early-out path).

## Placement summary

| Piece | Home |
|---|---|
| `domain_descriptor` struct | `cut/` |
| `make_domain_descriptor` | `cut/` |
| `resolve_domain_partition` core | `topology/domains/` |
| extended `make_nesting_merges` (WantInclusion) | `topology/domains/` |
| generalized `propagate_inclusion_bits` overload | `cut/arrangements/` |
| generalized `anchor_sheet_sides` | `cut/arrangements/` |

The CSG graph is untouched; it keeps its own richer internal descriptor
and simply shares the generalized `propagate`/`anchor` cores.
