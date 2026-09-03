# Triangulated Face Cuts — CSG Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the slit / partial-wall domain collapse in the CSG pipeline by triangulating cut loops *before* arrangement connectivity is built, via a new csg-only `triangulated_face_cuts` that replaces `face_cuts` in the `csg_graph` path.

**Architecture:** A single closed surface whose cut leaves a free edge inside a closed volume produces a *slit*: one region loop visits the slit tip twice (`…h,tip,h…`). On region loops, connectivity collapses the two slit walls into one incidence (K=3 fan) → the closed surface's two sides merge → `n_domains` collapses. Triangulating the loops first makes the two walls distinct triangles (K=4 fan), resolving it. `arrangement_graph` is csg-only, so we change its loop input type rather than keep `face_cuts` compatibility. Flat ids are computed once in the new class and reused by `arrangement_graph` (no re-flatten). Coplanar dedup is split: whole coincident walls deduped at the **region** level in the class (Delaunay can triangulate two coincident regions differently on a cocircular tie, so triangle-level matching is unsafe for whole walls); additional partial-overlap pairs are added at the **triangle** level in `arrangement_graph`.

**Tech Stack:** Header-only C++17, trueform `snake_case`, trailing return types, `make_` factories, leading-underscore privates, TBB-backed parallel primitives, `generic_generate` with reusable state (no allocation in hot loops).

## Global Constraints

- The new class and the `arrangement_graph` change are **csg-only**. `face_cuts`, `cut_graph`, `make_boolean`, and the iso / mesh-arrangement pipelines keep the region `face_cuts` — do not touch them. In particular **do not touch `cut_graph::build_flat_ids`** — it is algorithmically identical to the class's flatten but is the non-csg boolean path.
- **Region loops are fully transient.** The cutter (`face_cuts::build`, same cost as today) runs once inside the class; its region loops feed only the dedup + triangulation, then are discarded. No downstream consumer reads region geometry — the descriptor / `canonicalize_nm_edges` use the *original face corners* via `get_point`, and volumes/seeds use triangle vertices. Do not retain or re-expose region loops.
- No allocations in hot loops: per-element scratch goes through `generic_generate` state.
- Flat ids computed exactly once (in the class); `arrangement_graph` reuses them.
- Coplanar pair record is `std::array<Index,3> = {survivorTri, otherTag, reversed}` everywhere downstream — `[1]` is a **tag**, not a loop id.
- Triangulation projector must use `projection_axes(corner0,corner1,corner2)` of the original face — identical to `face_cuts::build` and extraction, so triangles match for reuse.
- Reference prototype: `experimentation/tri_face_cuts.cpp` (validated: slit edge → 4 incidences).

---

### Task 1: `triangulated_face_cuts` class in the library

**Files:**
- Create: `include/trueform/cut/triangulated_face_cuts.hpp`
- Test: `tests/cut/test_triangulated_face_cuts.cpp` (+ register in `tests/cut/CMakeLists.txt`)

**Interfaces:**
- Produces accessors: `loops()` → `make_blocked_range<3>` of `vertex_t`; `descriptors()`; `tag_offsets()`; `flat_triangles()` → `make_faces<3>` over flat ids; `n_flat()`; `coplanar_stack()` → range of `{survivorTri, otherTag, reversed}`.
- `build(ig, n_ipts, apply_to_face, get_mesh_point, get_point)`.

**Members kept:** `_tri_data` (`buffer<vertex_t>`, 3/triangle), `_flat_tri` (`buffer<Index>`, 3/triangle), `_descriptors` (`buffer<face_descriptor>`), `_tag_offsets`, `_n_flat`, `_coplanar_stack` (`buffer<std::array<Index,3>>`).

**Build pipeline (port from prototype, with region dedup + fan-out):**
1. Transient region `face_cuts fc; fc.build(ig, apply_to_face, get_mesh_point);` — empty → `_init_empty(n_tags)`.
2. `build_flat_ids(fc, n_ipts, flat_data, _n_flat)` (the prototype's verbatim copy of `cut_graph::build_flat_ids`). `flat_data` transient.
3. `_dedup_coplanar(fc, flat_data, _n_flat, region_pairs, dead)` — transient region `face_membership` + edge-0 `face_edge_neighbors` (small_vector via `generic_generate` state) + `compare_faces(with_tag(...))`. Emits **region** pairs `{survivorRegion, deadRegion, reversed}` and the `dead` region mask.
4. `_triangulate_survivors(...)` — `generic_generate` over `enumerate(zip(descs, loops))`, skip `dead[lid]`; emit per triangle: `vertex_t` triple → `_tri_data`, flat ids `flat_data[base+id]` → `_flat_tri`, descriptor → `_descriptors`, tag → `tri_tag`. **Also record, per surviving region, its triangle range** (contiguous because `generic_generate` is sequenced) into a transient `region_tri_offsets`.
5. `_build_tag_offsets(tri_tag, n_tags)`.
6. `_fan_out_coplanar(region_pairs, region_tri_offsets, fc.descriptors())` → for each region pair, for each survivor triangle `T` in the survivor region's range, push `{T, descs[deadRegion].tag, reversed}` into `_coplanar_stack`. (Do **not** sort here — `arrangement_graph` extends then sorts.)

**Do NOT build connectivity here** — `arrangement_graph` builds it from `flat_triangles()` + `n_flat()`.

- [ ] **Step 1: Write failing test** — build the sphere + partial-plane scene (translate `make_plane_mesh<Index>(2,4)` to x=-1), `make_csg_graph`, then build the class via `ig`/converter callables; assert the slit edge (a region loop's repeated-vertex tip & hinge) appears in exactly **4** triangles.

```cpp
// dynamic slit detection: find a region face_cuts loop visiting a vertex
// twice (h,tip,h); after class build, count triangles containing both h,tip.
ASSERT_EQ(count_edge_incidence(tfc, hinge, tip), 4);
```

- [ ] **Step 2: Run, verify FAIL** (`triangulated_face_cuts.hpp` not present).
- [ ] **Step 3: Implement** the header per the pipeline above. Methods with trailing return types; `_dedup_coplanar` passes `small_vector<Index,8>{}` as `generic_generate` state.
- [ ] **Step 4: Run, verify PASS** (4 incidences).
- [ ] **Step 5: Commit** `feat(cut): triangulated_face_cuts (csg-only) — region dedup + triangulate, keeps flat ids`.

---

### Task 2: Template the descriptor / csg consumers on the fc type

**Files (modify — change `const tf::face_cuts<Index,Int>&` parameter to a deduced `const FaceCuts&` template param; bodies use only `loops()/descriptors()/tag_offsets()` so they are unchanged):**
- `include/trueform/cut/arrangements/make_arrangement_descriptor.hpp`
- `include/trueform/cut/arrangements/canonicalize_nm_edges.hpp`
- `include/trueform/cut/arrangements/make_non_manifold_edge_fans.hpp`
- `include/trueform/cut/arrangements/emit_domain_merges.hpp`
- `include/trueform/cut/arrangements/compute_bundle_tag_index.hpp`
- `include/trueform/csg/graph/compute_arrangement_domain_volumes.hpp`
- `include/trueform/csg/graph/compute_bundle_aabbs.hpp`
- `include/trueform/csg/graph/seed_inclusion_bits.hpp`
- `include/trueform/csg/graph/make_csg_map_data.hpp`
- `include/trueform/csg/graph/make_csg_partition.hpp`
- `include/trueform/csg/graph/make_csg_domain_partition.hpp`

**Interfaces:**
- Consumes: any type exposing `loops()` (range of vertex ranges), `descriptors()`, `tag_offsets()`.
- Produces: identical behavior; callers pass either `face_cuts` (existing tests) or `triangulated_face_cuts`.

**Verified mechanical (audit):** all 13 functions use **only** `loops()` / `descriptors()` / `tag_offsets()` on the `fc` arg — none touch `loops_buffer()` / `offsets_buffer()` / `data_buffer()`. So this task is signature-only: change the parameter type to a deduced template, **no body edits**. (`arrangement_graph` is the *sole* place reaching into `loops_buffer()` — handled in Task 3.) `compute_arrangement_domain_volumes`' per-loop fan runs once per triangle and stays correct.

Pattern (apply to each signature):

```cpp
// before
template <typename Index, typename Int, ...>
auto f(..., const tf::face_cuts<Index, Int> &fc, ...) -> ...;
// after
template <typename Index, typename FaceCuts, ...>
auto f(..., const FaceCuts &fc, ...) -> ...;
```

Where a body constructs `tf::point<Int,...>` it already gets `Int` from `get_point`/callables, not from `fc`; keep those template params. `make_non_manifold_edge_fans` returns `non_manifold_edge_fans<Index>` keyed on `vertex_t` — unchanged (triangles still yield `vertex_t` loops).

- [ ] **Step 1:** Template each signature; remove the now-unused `#include "../face_cuts.hpp"` only where nothing else needs it (leave if other code does).
- [ ] **Step 2:** Build the existing csg tests (`tests/csg`) — must still compile & pass with `face_cuts` (no behavior change yet).
- [ ] **Step 3: Commit** `refactor(cut): template descriptor/csg consumers on the face-cuts type`.

---

### Task 3: Rewrite `arrangement_graph::build` for `triangulated_face_cuts`

**Files:**
- Modify: `include/trueform/cut/arrangement_graph.hpp`

**Interfaces:**
- Consumes: `triangulated_face_cuts` — `flat_triangles()`, `n_flat()`, `loops()`, `descriptors()`, `tag_offsets()`, `coplanar_stack()`.
- Produces: unchanged accessors — `connectivity_per_face_edge()`, `loop_labels()`, `polygon_labels()`, `coplanar_pairs()` (now `{survivorTri, otherTag, reversed}`), `open_component_mask()`, etc.

Changes:
1. `build(ig, tfc, forms)` template param becomes the fc type (or concrete `triangulated_face_cuts`).
2. **`_compute_loop_connectivity`**: drop the entire flatten block (per-tag maps, `flat_data`, `tag_base`). Replace with:

```cpp
auto flat_faces = tfc.flat_triangles();              // make_faces<3> over flat ids
tf::face_membership<Index> fm;
fm.build(flat_faces, tfc.n_flat(), /*incidences*/ 3 * n_tris);
tf::topology::compute_face_link_per_edge(flat_faces, fm,
    _edge_neighbour_offsets, _neighbours);
_loop_edge_offsets = stride-3 (or store none and group via make_blocked_range<3>);
```

3. **`_detect_coplanar_dead_loops`** runs on triangle loops (edge-0 + `compare_faces` unchanged). **Rewrite its emission in place** to push our shape `{survivorTri, descs[deadTri].tag, reversed}` directly — not `{survivorTri, deadTri, reversed}`. The dead-triangle mask no longer comes from `pair[1]` (now a tag); set `dead[deadTri]` from the emit (idempotent inline write, or a parallel dead-id buffer filled alongside the pairs). Seed `_coplanar_pairs` from `tfc.coplanar_stack()` (already our shape), append these, then `tbb::parallel_sort` + `std::unique`. Dead triangles → `_clean_connectivity` (existing).
4. CCL, `_compute_surface_labels`, `_collect_bridges`, `_compute_open_components`, id composition: unchanged (operate on `tfc.loops()` / connectivity — `resolve_face_edge` already skips diagonals, verified).

- [ ] **Step 1: Write failing test** — feed `triangulated_face_cuts` for the partial-wall scene through `arrangement_graph::build` + `make_arrangement_descriptor`; assert `desc.n_domains == 2` (sphere stays split).
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** the connectivity reuse + coplanar extend/sort/unique.
- [ ] **Step 4: Run, verify PASS** (`n_domains == 2`).
- [ ] **Step 5: Commit** `feat(cut): arrangement_graph consumes triangulated_face_cuts; reuse flat ids, extend coplanar stack`.

---

### Task 4: Coplanar consumers read the tag from the pair

**Files:**
- Modify: `include/trueform/cut/arrangements/compute_domain_inclusions.hpp`
- Modify: `include/trueform/cut/arrangements/propagate_inclusion_bits.hpp`

Changes (also template the fc param per Task 2 pattern):
- `compute_domain_inclusions.hpp:137`: `descs[dead].tag` → `(*it)[1]`. Drop the now-unused `dead` local.
- `propagate_inclusion_bits.hpp:84,88`: `const Index dead = p[1]; ... descs[dead].tag` → `const Index t = p[1];`.

- [ ] **Step 1:** Apply both one-line changes.
- [ ] **Step 2:** Build; existing csg tests still pass (with `face_cuts` path the pairs are still `{survivor,dead,rev}` — so this task lands together with Task 5/3 where the format flips; gate by running the full csg suite after Task 5).
- [ ] **Step 3: Commit** `refactor(cut): inclusion consumers read coplanar tag from pair[1]`.

---

### Task 5: Wire `csg_graph` to `triangulated_face_cuts`

**Files:**
- Modify: `include/trueform/csg/csg_graph.hpp`

Changes:
- Member `_fc` type: `tf::face_cuts<index_type, resolved_int_type>` → `tf::cut::triangulated_face_cuts<index_type, resolved_int_type>`.
- Ctor: replace `_fc.build(_ig, apply_to_face, get_mesh_point);` then `_ag.build(_ig, _fc, tagged);` with `_fc.build(_ig, n_ipts, apply_to_face, get_mesh_point, get_point);` then `_ag.build(_ig, _fc, tagged);` (`get_point` already constructed below — hoist it above the `_fc.build`).
- `face_cuts()` accessor return type → the new class.
- `#include "../cut/triangulated_face_cuts.hpp"`.

- [ ] **Step 1: Write failing test** — end-to-end `make_csg_domains` on the partial wall: assert it returns **2** bounded domains (was 1).
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** the ctor + member swap.
- [ ] **Step 4: Run, verify PASS**, then run the **entire** `tests/csg` suite (parity, nesting, sheets) — all green.
- [ ] **Step 5: Commit** `feat(csg): csg_graph builds triangulated_face_cuts (slit/partial-wall fix)`.

---

### Task 6: Drop re-triangulation in extraction

**Files:**
- Modify: `include/trueform/csg/graph/make_csg_mesh.hpp`
- Modify: `include/trueform/csg/graph/make_csg_domains.hpp`

The cut loops are already triangles. Stage 3 (`triangulate_partition_cuts` + `make_projector`) re-runs Delaunay on the same loops — delete it and gather the pre-made triangles for the selected `pids[t].cut_faces[L]` directly, mapping vertices via `map_data.map_vertex`. Each selected loop is exactly 3 vertices → emit `map_vertex(tag, loop[0..2])`.

- [ ] **Step 1: Write failing test** — none new; this is a perf/cleanup task gated by output equivalence: assert `make_csg_mesh`/`make_csg_domains` output for a corpus pair is unchanged (same triangle count & a watertightness/oracle check already in the suite).
- [ ] **Step 2:** Replace Stage 3 with a gather over the class's triangles.
- [ ] **Step 3:** Run the csg suite + a corpus spot-check; outputs equivalent.
- [ ] **Step 4: Commit** `perf(csg): extraction reuses pre-triangulated cut loops (drop re-triangulation)`.

---

### Task 7: Validation & regression

**Files:** none (or a benchmark under `experimentation/`, not committed).

- [ ] Partial-wall, two perpendicular partial walls, sphere-only: correct `n_domains`.
- [ ] Full `tests/csg` suite green (boolean parity, domains parity vs topology oracle, nesting, sheets).
- [ ] Corpus regression: `make_csg_domains` / `make_csg_mesh` over `corpus_large` — validity oracle unchanged vs `main`.
- [ ] Perf: confirm the build delta is ~the benchmarked net (triangulation offset by dropped extraction re-triangulation); record numbers.

---

## Self-Review Notes

- **Coplanar correctness** hinges on Task 3's extend-sort-unique producing `{survivorTri, otherTag, reversed}` for *both* region (Task 1 fan-out) and triangle (Task 3) sources. Verify a two-cube shared-wall case: the domain bounded by the shared wall must carry *both* operand tags.
- **`n_flat` / incidence count** for `fm.build` in Task 3: incidences = `_flat_tri.size()` = `3 * n_tris`; vertices = `tfc.n_flat()`.
- **Type ripple**: Task 2 must land before Tasks 3–5 or the tree won't compile mid-way; Task 4's format flip is only correct once Task 3/5 emit the new pair format — run the full suite after Task 5, not Task 4.
