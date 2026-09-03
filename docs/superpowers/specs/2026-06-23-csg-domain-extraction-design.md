# CSG domain extraction + source-ignore — Design

**Goal:** Extend the N-ary CSG-graph extraction so the same `csg_graph` + boolean
expression can be read two ways — as a merged boolean **mesh** (today) or as
per-domain watertight **cells** (new) — and add an optional **source filter** to the
mesh read so you can keep only specific operands' surfaces.

**Scope of this spec:** primarily `make_csg_domains`; `make_csg_mesh` gains an optional
`ignore`. Python/TS bindings are a follow-on.

---

## Model — two orthogonal axes

Every extraction is a point in 2D:

1. **Region** — *which domains*: a `tf::csg::expr` over per-domain inclusion bits
   (`inside(i)=op(i)`, `outside(i)=~op(i)`, composed with `merge/intersection/
   difference/complement`). Evaluated by `evaluate_per_domain(inclusion, E)` → bool
   per domain. Already shared by both reads.
2. **Source** — *whose surface* to emit: which operands' faces appear in the output.
   Only meaningful for the **mesh** read (a merged solid's boundary can come from any
   operand). Expressed as an optional `ignore` keep-mask.

| want | call | region | source |
|---|---|---|---|
| boolean solid | `make_csg_mesh(g, op(0) & ~op(1))` | expr | all (capped) |
| A's surface one side of plane P | `make_csg_mesh(g, side_of_P, ignore={P})` | expr | only A (open) |
| all bounded cells | `make_csg_domains(g)` | select-all | n/a |
| cells outside form 2 | `make_csg_domains(g, ~op(2))` | expr | n/a |

The **outer/unbounded domain** is the all-zero inclusion bitvector. `merge`/`any_of`
already exclude it; `complement`/`difference` would select it, so extraction excludes
the outer/unbounded domain by default (drop all-zero-bit domains).

---

## API

Two extraction entry points on a built `csg_graph`, both with empty-range forwarding
(mirroring `make_csg_graph(forms, sheets=∅)`):

```cpp
// mesh read — merged boolean solid; optional source keep-mask
auto make_csg_mesh(graph, expr);                       // ignore = ∅ (keep all)
auto make_csg_mesh(graph, expr, ignore_range);         // drop faces whose source ∈ ignore

// domain read — per-domain watertight cells; NO ignore (cells need all walls)
auto make_csg_domains(graph, expr);                    // cells where expr(domain) is true
auto make_csg_domains(graph);                          // all bounded domains (no expr = every domain)
```

`ignore_range` is a range of operand ids; impl checks `size()==0` → skip the filter
(one impl, no overload explosion). `make_csg_domains` has **no** ignore.

**Naming/structure:** `make_csg_domains` mirrors `make_csg_mesh` — thin public wrapper
in `csg/`, heavy impl in `csg/graph/`. No expression ⇒ all bounded domains; the
expression only narrows.

**Implementation note (no materialisation):** `make_csg_domains` works on the **implicit
graph**, exactly like `make_csg_mesh` — it does **not** build a full arrangement mesh and
call `tf::split_into_domains`. It is a fusion of `split_into_domains`' per-domain emission
*pattern* (argsort by domain → offset blocks → reused `point_map` watermark; side-0
reversed) with `make_csg_mesh`'s implicit-graph machinery (`make_csg_map_data`,
`triangulate_partition_cuts`). **Performance must match `make_csg_mesh`** — fully parallel
emission, no serial heavy loops.

**Returns:**
- `make_csg_mesh`: one `polygons_buffer` (unchanged) — but now also a **per-face source
  tag** stream (needed for `ignore`; useful output regardless), matching the 2-op
  `make_boolean`'s `labels`.
- `make_csg_domains`: per-domain meshes + the domain id per mesh (+ each domain's
  inclusion bitvector, so the caller knows what each cell is inside) — the
  `split_into_domains` return shape, lifted to the implicit graph.

---

## Implementation

### `make_csg_domains` (primary)

Reuse ~90% of `csg/graph/make_csg_mesh.hpp`. The geometry assembly
(`make_csg_map_data` vertex discovery/dedup, `triangulate_partition_cuts`, point
materialisation) is already label-count-agnostic; `make_partition_ids` counting-sorts
over any K labels. What changes is the **selection/labeling** and the **emission**:

1. `evaluate_per_domain(graph.inclusion(), E)` → `membership[d]` (which domains to keep);
   for `make_csg_domains(graph)` membership = "domain is bounded" (bit-set, i.e. not the
   outer all-zero domain).
2. New `compute_domain_partition(desc, membership)`: per component `c`, for each side
   `s∈{0,1}` whose `domain_of_side[2c+s]` is a kept domain, emit a label = that domain
   id with direction = (`s==0` → reverse, `s==1` → forward). This is the
   `split_into_domains` side→domain→winding pattern (`split_into_domains.hpp:73-135`)
   lifted onto `desc.domain_of_side` instead of a materialised `domain_labels`. A wall
   between two kept domains is emitted **twice** (once per side) — correct: separate
   watertight cells.
3. Generalise the emission in `make_csg_mesh.hpp:109-189` from the hardwired 2-label
   `{rev,fwd}` (the `std::array<...,2>` + 4-stream concat) to **K = number of kept
   domains**, each domain getting its own output block (one `polygons_buffer` per
   domain). Per-domain triangulation/remap reuse the existing per-(form,label) machinery
   with the domain-id as the label.
4. Outer/sentinel domains are not in `membership`, so they never emit (mirrors
   `split_into_domains`' sentinel-trim).

### `make_csg_mesh` `ignore` (secondary, the source axis)

1. Emit a **per-face source tag** from `make_csg_mesh` (each output face already comes
   from a known form `t` in the assembly loop — record it; the swarm-tagged experiment
   `csg_bunny_swarm_tagged.cpp` shows the exact spot). This is independently useful.
2. With a non-empty `ignore`, post-filter the assembled faces: drop any whose source tag
   ∈ ignore. (Cheap stream compaction after assembly.) Empty `ignore` → no-op.

### Reuse map (from the pipeline study)
- Reuse as-is: `make_csg_map_data`, `triangulate_partition_cuts`, point dedup,
  `make_partition_ids` (K-label generic).
- New: `compute_domain_partition` (sibling of `compute_chosen_sides`), the K-block
  emission generalisation, the per-face source tag, the `make_csg_domains` entry +
  empty-range `ignore` overload.
- Not reused: `split_into_domains` itself (operates on a materialised mesh, not the
  implicit graph) — only its selection *pattern*.

---

## Correctness / edge cases

- **Watertightness:** each kept domain emits every side that maps to it, oriented
  outward (side 0 reversed, side 1 forward) — same convention as `compute_chosen_sides`
  / `split_into_domains`, so each cell is closed.
- **Shared walls duplicate** across the two cells (by design); the union of cells is
  intentionally non-manifold at shared walls.
- **Sheets:** `inside(sheet)/outside(sheet)` already work via the bit ("behind the
  normal"); no special handling. A sheet's two sides are distinct domains already.
- **Coplanar/aligned:** handled upstream in the arrangement; the domain-id labeling is
  agnostic to it.
- **Empty selection** (no domain matches expr) → empty result, not an error.

## Testing

- `make_csg_domains(graph)` on a closed mesh cut by N planes reproduces the interior cells
  we render today (count + total volume == the topology-path `split_into_domains`).
- `make_csg_domains(graph, op(i))` → form i's pieces; merging them == `make_csg_mesh(g,
  op(i))` (same surface, watertight).
- Plane-cut: `make_csg_mesh(g, side_of_P, ignore={P})` → A's open surface piece (no cap),
  and the two sides reassemble to A's full surface.
- Each cell closed + manifold; shared-wall duplication verified.

## Deferred
- Python/TS bindings (`make_csg_domains`, the `ignore` arg).
- `from(S)` sugar over `ignore` (dual); not needed for v1.
- Per-domain provenance tags inside cells (cells already carry the inclusion bitvector).
