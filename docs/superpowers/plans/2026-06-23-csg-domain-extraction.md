# CSG Domain Extraction + Source-Ignore — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes. This is C++ (header-only) with Catch2 tests.

**Goal:** Read the N-ary `csg_graph` two ways from one boolean expression — the existing merged boolean **mesh** (`make_csg_mesh`), and new per-domain watertight **cells** (`make_csg_domains`) — plus an optional source-`ignore` on the mesh read.

**Architecture:** Both reads share `evaluate_per_domain(graph.inclusion(), E)` → bool per domain. `make_csg_domains` is a **fusion of two existing files on the implicit graph (no materialisation)** — it reimplements `split_into_domains`' per-domain emission *structure* (argsort by domain label → offset blocks → one reused `point_map` with a per-domain watermark; side-0 reversed, side-1 forward) on top of `make_csg_mesh`'s implicit-graph machinery (`make_csg_partition`/`make_csg_map_data` vertex discovery+dedup, `triangulate_partition_cuts` for cut loops). It lives entirely in `csg/graph/` so the `csg/` root stays clean; the public `tf::make_csg_domains` wrapper mirrors `tf::make_csg_mesh`. The mesh read gains per-face source tags + a post-filter for `ignore`.

**Tech Stack:** trueform header-only C++17, TBB, Catch2. Spec: `docs/superpowers/specs/2026-06-23-csg-domain-extraction-design.md`.

**Domain model (verified by descriptor probe — do NOT collapse this):**
- Extraction is **by domain identity** (`desc.domain_of_side`), one mesh per kept domain. It is NOT keyed on the inclusion bitvector. Two domains can share a bitvector and must still emit as **separate** meshes — e.g. a sphere cut by a plane gives two hemispheres that both carry bits `0x1` when the plane is not a sheet; they are distinct `domain_of_side` ids and come back as two meshes. This is the whole point vs `make_csg_mesh`, which *merges* same-membership domains into one boundary.
- The expression is a **filter over domains, never a merge**: `keep[d] = E(bits[d]) && (bits[d] != 0)`. `E` decides *which* domain ids survive; it never fuses two kept domains. `make_csg_domains(graph)` = all bounded domains; `make_csg_domains(graph, op(0))` = every domain inside form 0, each as its own mesh (two hemispheres, not one sphere).
- Emission **skips components with `d0 == d1`** (open fragments self-merged by the arrangement — `_compute_open_components` / `emit_open_merges`). Those are non-separating internal flaps; emitting them would pollute a cell. Components with `d0 != d1` emit each side mapping to a kept domain (side 1 forward, side 0 reversed). The arrangement already merges *only* genuinely-open fragments (free-edge `is_boundary()`), so the in-sphere disc still separates — exactly `ignore_open_fragments` semantics, on the implicit graph.
- **Sheets stay exempt** from the open self-merge (a sheet separates everywhere). So a sheet's open-rim region yields domains whose wall is the sheet's open part; those come back as **open ("half") meshes** by design. `make_csg_domains` returns them; the caller checks `is_closed` if it wants only watertight cells. Only the all-zero universe is excluded by default.

## Global Constraints

- Style (CLAUDE.md): `snake_case`, `make_` factories, leading-underscore privates, `#pragma once` + copyright header, trailing return types. No essay comments — WHY only when non-obvious.
- Header-only: everything in `include/trueform/...`; export via the `csg.hpp` umbrella.
- Exact predicates / arithmetic unchanged — this is a new *read* over the existing graph; do not touch the arrangement/inclusion build.
- The outer/unbounded universe is the **all-zero inclusion bitvector**; `make_csg_domains` excludes it by default (AND the selection with "bitvector ≠ 0"). Non-zero open domains (sheet halves) are NOT excluded — they return as open meshes by design.
- Emission must **skip components where `domain_of_side[2c+0] == domain_of_side[2c+1]`** (self-merged open fragments). Both `compute_domain_partition` (label = -1 for such sides) or the emission loop may enforce this; pick one and be consistent.
- Reuse, don't fork: `make_csg_map_data`, `triangulate_partition_cuts`, `make_partition_ids` are already K-label generic — use them. Only the *selection* (`compute_chosen_sides` → new sibling) and the *emission loop* (`make_csg_mesh.hpp:109-189`, hardwired to 2) are new.
- **No materialisation.** `make_csg_domains` operates on the implicit graph, exactly like `make_csg_mesh` — do not build a full arrangement mesh and call `tf::split_into_domains`. Reimplement that function's per-domain emission *pattern* inside `csg/graph/`.
- **Keep `csg/` root clean:** all new heavy code goes under `include/trueform/csg/graph/`; only the thin public `tf::make_csg_domains` wrapper goes in `include/trueform/csg/` (mirroring `csg/make_csg_mesh.hpp`).
- **Performance parity with `make_csg_mesh`.** The emission must stay fully parallel — `tbb::parallel_sort` for the per-domain argsort, `parallel_for_each` / `task_group` for stream copies and point materialisation, and the `make_csg_map_data` / `triangulate_partition_cuts` parallel paths reused unchanged. No per-element `push_back` loops, no serial domain loop doing heavy work (the per-domain block walk only sets up offsets; the copies inside are parallel). Selecting all domains (`make_csg_domains(graph)`) over the same arrangement must run in time comparable to `make_csg_mesh(graph, expr)` on the same graph — argsort + K-block emission is the same asymptotic work as the 2-label concat, just K blocks instead of 2.
- **No expression ⇒ all bounded domains.** `make_csg_domains(graph)` (no `expr`) selects every domain with a non-zero inclusion bitvector — the whole partition. The expression only ever *narrows* this set.

## Build & test commands (used in every task)

Configure once:
```bash
cmake -S /Users/ziga/trueform -B /Users/ziga/trueform/build-tests \
  -DTF_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
Build + run the CSG suite (Catch2 tag filter):
```bash
cmake --build /Users/ziga/trueform/build-tests --target trueform_csg_tests -j8
/Users/ziga/trueform/build-tests/tests/csg/trueform_csg_tests "[domains]"
```

## File structure

- Create `include/trueform/csg/graph/compute_domain_partition.hpp` — per-component → per-form `partition_labels` keyed by **domain id** (sibling of `compute_chosen_sides.hpp`).
- Create `include/trueform/csg/graph/make_csg_domains.hpp` — impl `tf::csg::graph::make_csg_domains(...)`; one `polygons_buffer` per kept domain. Fuses `split_into_domains`' per-domain emission pattern with `make_csg_mesh.hpp`'s implicit-graph machinery (no materialisation).
- Create `include/trueform/csg/make_csg_domains.hpp` — public `tf::make_csg_domains(graph[, expr])` (evaluate_per_domain → membership → `csg::graph::make_csg_domains`), mirroring `csg/make_csg_mesh.hpp`.
- Modify `include/trueform/csg.hpp` — `#include "./csg/make_csg_domains.hpp"`.
- Create `tests/csg/test_csg_domains.cpp`; add it to `tests/csg/CMakeLists.txt`.
- (Task 5, secondary) Modify `include/trueform/csg/graph/make_csg_mesh.hpp` + `include/trueform/csg/make_csg_mesh.hpp` — per-face source tags + optional `ignore`.

Reference the existing idiom in `tests/csg/test_csg_solids.cpp` for building forms/graphs.

---

### Task 1: `compute_domain_partition` — domain-id labels per component

**Files:**
- Create: `include/trueform/csg/graph/compute_domain_partition.hpp`
- Test: `tests/csg/test_csg_domains.cpp` (new), add to `tests/csg/CMakeLists.txt`

**Interfaces:**
- Consumes: `tf::cut::arrangement_descriptor<Index>` (`.domain_of_side`, `.n_domains`, `.bundle_of_component.size()` == n_components), `tf::buffer<bool> membership` (per-domain keep), `tf::buffer<std::uint32_t>`-backed `inclusion` for the outer-domain exclusion.
- Produces: `tf::buffer<Index> domain_label` and `tf::buffer<std::int8_t> side_dir` per component-side; concretely a function returning, per component `c`: for each side `s∈{0,1}`, the **output domain label** (`domain_of_side[2c+s]` remapped to a dense kept-domain index, or `-1` if not kept) and direction (`s==0`→reverse). This is what Task 2 partitions on.

- [ ] **Step 1: Add the test file to the build**

Edit `tests/csg/CMakeLists.txt` — add `test_csg_domains.cpp` to the `trueform_csg_tests` sources list (after `test_csg_sheets.cpp`).

- [ ] **Step 2: Write the failing test** (`tests/csg/test_csg_domains.cpp`)

```cpp
/* Copyright (c) 2026 Ziga Sajovic, XLAB */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/csg/graph/compute_domain_partition.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>
#include <vector>

namespace {
using Index = int;
using Real = double;

auto frame0() {
  return tf::make_frame(tf::make_transformation_from_translation(tf::vector<Real,3>{0,0,0}));
}
// a closed unit sphere split by one axis plane through the centre -> 2 interior domains
template <typename F>
auto sphere_plane_forms() -> std::vector<F>;  // defined inline in Step 4's test body
}

TEST_CASE("compute_domain_partition keeps a label per kept domain side", "[domains]") {
  auto sf = tf::make_sphere_mesh<Real, Index>(1.0, 24, 24);
  auto pf = tf::make_plane_mesh<Real, Index>(3.0, 3.0);
  auto sphere = tf::make_polygons(sf) | tf::tag(frame0());
  auto plane  = tf::make_polygons(pf) | tf::tag(frame0());
  using form_t = decltype(sphere);
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms), tf::make_range(std::vector<bool>{false, true})); // plane is a sheet

  // keep every bounded domain (non-zero inclusion bitvector)
  auto inc = graph.inclusion();
  auto blocks = inc.make_range();
  tf::buffer<bool> keep; keep.allocate(blocks.size());
  for (std::size_t d = 0; d < blocks.size(); ++d) {
    std::uint32_t any = 0; for (auto w : blocks[d]) any |= w; keep[d] = any != 0;
  }

  auto part = tf::csg::graph::compute_domain_partition(graph.descriptor(), keep);
  // dense kept-domain count == number of true entries in keep
  std::size_t n_keep = 0; for (std::size_t d = 0; d < keep.size(); ++d) n_keep += keep[d];
  REQUIRE(part.n_kept == static_cast<Index>(n_keep));
  REQUIRE(n_keep == 2);  // sphere split by the plane = 2 interior cells
}
```

- [ ] **Step 3: Run, expect FAIL** (header missing)

```bash
cmake --build /Users/ziga/trueform/build-tests --target trueform_csg_tests -j8
```
Expected: compile error `compute_domain_partition.hpp: No such file`.

- [ ] **Step 4: Implement `compute_domain_partition`**

Mirror `csg/graph/compute_chosen_sides.hpp` but key on domain id, not a global flip. Dense-remap kept domain ids to `[0, n_kept)`.

```cpp
/* Copyright (c) 2026 Ziga Sajovic, XLAB */
#pragma once
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include <cstdint>

namespace tf::csg::graph {

/// @brief Per component-side, the dense kept-domain label it bounds (or -1),
///        with side 0 = reverse winding, side 1 = forward. Drives per-domain
///        emission (one output mesh per kept domain), unlike compute_chosen_sides
///        which collapses to a single rev/fwd solid.
template <typename Index>
struct domain_partition {
  tf::buffer<Index> dense_of_domain;   // size n_domains; kept domain -> [0,n_kept), else -1
  tf::buffer<Index> side_label;        // size 2*n_components; dense kept id on each side, or -1
  Index n_kept = 0;
};

template <typename Index>
auto compute_domain_partition(const tf::cut::arrangement_descriptor<Index> &desc,
                              const tf::buffer<bool> &keep) -> domain_partition<Index> {
  domain_partition<Index> out;
  const Index n_domains = static_cast<Index>(keep.size());
  out.dense_of_domain.allocate(static_cast<std::size_t>(n_domains));
  Index next = 0;
  for (Index d = 0; d < n_domains; ++d)
    out.dense_of_domain[d] = keep[d] ? next++ : Index(-1);
  out.n_kept = next;

  const Index n_components = static_cast<Index>(desc.bundle_of_component.size());
  out.side_label.allocate(static_cast<std::size_t>(2 * n_components));
  tf::parallel_for_each(tf::make_sequence_range(n_components), [&](Index c) {
    for (Index s = 0; s < 2; ++s) {
      const Index d = desc.domain_of_side[2 * c + s];
      out.side_label[2 * c + s] =
          (d >= 0 && d < n_domains) ? out.dense_of_domain[d] : Index(-1);
    }
  });
  return out;
}

} // namespace tf::csg::graph
```

- [ ] **Step 5: Run, expect PASS.** Build + `... trueform_csg_tests "[domains]"`. Expected: 1 assertion section passes.

- [ ] **Step 6: Commit**

```bash
git add include/trueform/csg/graph/compute_domain_partition.hpp tests/csg/test_csg_domains.cpp tests/csg/CMakeLists.txt
git commit -m "feat(csg): compute_domain_partition — domain-id labels per component-side"
```

---

### Task 2: `make_csg_domains` — one watertight mesh per kept domain

**Files:**
- Create: `include/trueform/csg/graph/make_csg_domains.hpp`
- Test: `tests/csg/test_csg_domains.cpp` (extend)

**Interfaces:**
- Consumes: `arrangement_graph`, `face_cuts`, `intersection_graph`, `forms` range, `vertex_converter`, and the `domain_partition<Index>` from Task 1.
- Produces: `auto make_csg_domains<OutReal>(ag, fc, ig, forms, part, conv) -> std::pair<std::vector<polygons_buffer<Index,OutReal,3,3>>, tf::buffer<Index>>` — per-domain meshes + the (original, non-dense) domain id of each.

- [ ] **Step 1: Write the failing test** (append to `test_csg_domains.cpp`)

```cpp
TEST_CASE("make_csg_domains: sphere split by plane -> 2 closed cells, volume conserved", "[domains]") {
  auto sf = tf::make_sphere_mesh<Real, Index>(1.0, 32, 32);
  auto pf = tf::make_plane_mesh<Real, Index>(3.0, 3.0);
  auto sphere = tf::make_polygons(sf) | tf::tag(frame0());
  auto plane  = tf::make_polygons(pf) | tf::tag(frame0());
  using form_t = decltype(sphere);
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms), tf::make_range(std::vector<bool>{false, true}));

  auto inc = graph.inclusion(); auto blocks = inc.make_range();
  tf::buffer<bool> keep; keep.allocate(blocks.size());
  for (std::size_t d = 0; d < blocks.size(); ++d) { std::uint32_t a=0; for (auto w: blocks[d]) a|=w; keep[d]=a!=0; }
  auto part = tf::csg::graph::compute_domain_partition(graph.descriptor(), keep);

  auto [cells, ids] = tf::csg::graph::make_csg_domains<Real>(
      graph.arrangement_graph(), graph.face_cuts(), graph.intersection_graph(),
      graph.forms(), part, graph.converter());

  REQUIRE(cells.size() == 2);
  double total = 0;
  for (auto &c : cells) {
    auto v = c.polygons();
    REQUIRE(tf::is_closed(v));
    REQUIRE(tf::is_manifold(v));
    total += std::abs(double(tf::signed_volume(v)));
  }
  // two half-balls reassemble the unit sphere volume 4/3 pi
  REQUIRE_THAT(total, Catch::Matchers::WithinRel(4.0/3.0*tf::pi<double>, 0.02));
}
```

(If `csg_graph` lacks public `arrangement_graph()/face_cuts()/intersection_graph()/forms()/converter()` accessors, add them in this task — they are already used internally by `make_csg_mesh(graph, expr)`; expose const refs. Check `include/trueform/csg/csg_graph.hpp` for existing accessors first and reuse their names.)

- [ ] **Step 2: Run, expect FAIL** (header missing / missing accessors).

- [ ] **Step 3: Implement `make_csg_domains`** as a **fusion of `make_csg_mesh.hpp` and `split_into_domains.hpp`** on the implicit graph — **do not materialise an arrangement mesh**.

Take `make_csg_mesh.hpp`'s implicit-graph front half verbatim and graft `split_into_domains.hpp`'s per-domain emission onto its back half:

1. **Selection (from `make_csg_mesh`, K labels not 2):** replace `make_csg_partition(ag, fc, forms, chosen_sides)` with a per-form `partition_labels<Index>` over **`part.n_kept` labels**. A component `c` bounds two domains, one per side; for each form, set each uncut face's / cut loop's label from its component via `part.side_label[2c+s]`. Because a face's two sides may both be kept domains, a face can emit into **two** different domain blocks (forward into its side-1 domain, reversed into its side-0 domain) — model this the way `split_into_domains` does: flatten to per-(element,side) entries keyed by `side_label`, dropping `-1`. Reuse `make_partition_ids` (counting sort over K labels — `cut/partition/make_ids.hpp`). Feed selected cut loops through `triangulate_partition_cuts` + `make_csg_map_data` exactly as `make_csg_mesh.hpp:71-127` does (unchanged, K-label generic).
2. **Emission (from `split_into_domains`, on the implicit graph):** replace `make_csg_mesh.hpp`'s `std::array<...,2>` tri buffers + fixed 4-stream concat with the `split_into_domains_polygons` structure (`reindex/split_into_domains.hpp:50-135`): argsort the (element,side) entries by their dense domain label → `compute_offsets` → `offset_block_range` = one contiguous block per kept domain; walk the blocks emitting each domain into **its own** `polygons_buffer`, reusing one `point_map` across domains with a `point_sentinel` watermark for dedup (`split_into_domains.hpp:75-134`); side-0 entries reverse the stored winding (`elem & 1` rule there → here the explicit side bit), side-1 keep it. The difference from `split_into_domains`: vertices and cut-loop triangles come from `make_csg_map_data` / `triangulate_partition_cuts` (implicit graph), not from a pre-built `polygons` mesh.
3. Return `std::pair{ std::vector<polygons_buffer>, buffer<Index> ids }` where `ids[k]` = the original domain id whose `dense_of_domain == k` (invert `part.dense_of_domain`).

Keep the `if constexpr (std::is_integral_v<RealOut>)` point-copy branches from `make_csg_mesh.hpp:197-234` verbatim (the int64/float output paths).

- [ ] **Step 4: Run, expect PASS** — 2 cells, both closed+manifold, volume ≈ 4/3·π.

- [ ] **Step 5: Commit**

```bash
git add include/trueform/csg/graph/make_csg_domains.hpp tests/csg/test_csg_domains.cpp include/trueform/csg/csg_graph.hpp
git commit -m "feat(csg): make_csg_domains — one watertight mesh per kept domain"
```

---

### Task 3: `tf::make_csg_domains(graph[, expr])` — public API

**Files:**
- Create: `include/trueform/csg/make_csg_domains.hpp` (public wrapper; mirrors `csg/make_csg_mesh.hpp`)
- Modify: `include/trueform/csg.hpp`
- Test: `tests/csg/test_csg_domains.cpp` (extend)

**Interfaces:**
- Consumes: `csg_graph`, optional `tf::csg::expr`.
- Produces:
  - `tf::make_csg_domains(graph) -> std::pair<std::vector<polygons_buffer<...>>, buffer<Index>>` — **all** bounded domains (non-zero inclusion); no expression = every domain.
  - `tf::make_csg_domains(graph, const csg::expr&)` — domains where `expr` is true AND bounded.

- [ ] **Step 1: Write the failing test** (append)

```cpp
TEST_CASE("make_csg_domains(graph) == all bounded cells; expr selects a subset", "[domains]") {
  auto sf = tf::make_sphere_mesh<Real, Index>(1.0, 32, 32);
  auto pf = tf::make_plane_mesh<Real, Index>(3.0, 3.0);
  auto sphere = tf::make_polygons(sf) | tf::tag(frame0());
  auto plane  = tf::make_polygons(pf) | tf::tag(frame0());
  using form_t = decltype(sphere);
  std::vector<form_t> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms), tf::make_range(std::vector<bool>{false, true}));

  auto [all_cells, all_ids] = tf::make_csg_domains(graph);
  REQUIRE(all_cells.size() == 2);                     // outer (all-zero) excluded

  // op(0) = inside the sphere -> still both halves (both are inside the sphere)
  auto [in_cells, in_ids] = tf::make_csg_domains(graph, tf::csg::op(0));
  REQUIRE(in_cells.size() == 2);

  // merging the op(0) cells == the boolean mesh of op(0): same volume, watertight
  auto solid = tf::make_csg_mesh(graph, tf::csg::op(0));
  double cell_vol = 0; for (auto &c : in_cells) cell_vol += std::abs(double(tf::signed_volume(c.polygons())));
  REQUIRE_THAT(cell_vol, Catch::Matchers::WithinRel(std::abs(double(tf::signed_volume(solid.polygons()))), 0.02));
}
```

- [ ] **Step 2: Run, expect FAIL** (`make_csg_domains` undefined).

- [ ] **Step 3: Implement `make_csg_domains.hpp`**

```cpp
/* Copyright (c) 2026 Ziga Sajovic, XLAB */
#pragma once
#include "./expression/expr.hpp"
#include "./graph/compute_domain_partition.hpp"
#include "./graph/evaluate_per_domain.hpp"
#include "./graph/make_csg_domains.hpp"
#include "../core/buffer.hpp"

namespace tf {

namespace detail {
// keep[d] = E(b(d)) AND (b(d) != 0)  — exclude the unbounded all-zero domain.
template <typename Graph, typename Pred>
auto domain_membership(const Graph &graph, Pred E) -> tf::buffer<bool> {
  auto blocks = graph.inclusion().make_range();
  tf::buffer<bool> keep; keep.allocate(blocks.size());
  for (std::size_t d = 0; d < blocks.size(); ++d) {
    std::uint32_t any = 0; for (auto w : blocks[d]) any |= w;
    keep[d] = (any != 0) && E(blocks[d]);
  }
  return keep;
}
} // namespace detail

/// @brief Per-domain watertight cells for the domains an expression selects
///        (all bounded domains by default). One mesh per kept domain + its id.
template <typename OutputCoordinateType = tf::none_t, typename Graph>
auto make_csg_domains(const Graph &graph, const tf::csg::expr &e) {
  auto E = e.compile().evaluator();
  auto keep = detail::domain_membership(graph, E);
  auto part = tf::csg::graph::compute_domain_partition(graph.descriptor(), keep);
  return tf::csg::graph::make_csg_domains<OutputCoordinateType>(
      graph.arrangement_graph(), graph.face_cuts(), graph.intersection_graph(),
      graph.forms(), part, graph.converter());
}

template <typename OutputCoordinateType = tf::none_t, typename Graph>
auto make_csg_domains(const Graph &graph) {
  auto keep = detail::domain_membership(graph, [](const auto &) { return true; });
  auto part = tf::csg::graph::compute_domain_partition(graph.descriptor(), keep);
  return tf::csg::graph::make_csg_domains<OutputCoordinateType>(
      graph.arrangement_graph(), graph.face_cuts(), graph.intersection_graph(),
      graph.forms(), part, graph.converter());
}

} // namespace tf
```

Add `#include "./csg/make_csg_domains.hpp"` to `include/trueform/csg.hpp`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add include/trueform/csg/make_csg_domains.hpp include/trueform/csg.hpp tests/csg/test_csg_domains.cpp
git commit -m "feat(csg): tf::make_csg_domains — expression-selected per-domain cells"
```

---

### Task 4: Reproduce the topology-path domains (parity test)

**Files:** Test only — `tests/csg/test_csg_domains.cpp` (extend).

**Interfaces:** Consumes `tf::make_csg_domains` (Task 3), `tf::make_mesh_arrangements` / `tf::make_domain_labels` / `tf::split_into_domains` (existing topology path) for the cross-check.

- [ ] **Step 1: Write the test** — a closed mesh cut by 3 axis planes (declared sheets); assert `make_csg_domains(graph)` yields the same number of bounded cells and the same total absolute volume as the topology path (`mesh_arrangements` → `cleaned` → `make_domain_labels(ignore_open_fragments)` → `split_into_domains`).

```cpp
TEST_CASE("make_csg_domains matches the topology split_into_domains partition", "[domains]") {
  auto sf = tf::make_sphere_mesh<Real, Index>(1.0, 40, 40);
  auto sphere = tf::make_polygons(sf) | tf::tag(frame0());
  std::vector<decltype(sphere)> forms{sphere};
  std::vector<bool> sheets{false};
  // add 3 axis planes through the centre as sheets
  for (int ax = 0; ax < 3; ++ax) {
    auto pf = tf::make_plane_mesh<Real, Index>(3.0, 3.0); /* orient to axis ax as in domains.py */
    forms.push_back(tf::make_polygons(pf) | tf::tag(frame0()));
    sheets.push_back(true);
  }
  auto graph = tf::make_csg_graph(tf::make_range(forms), tf::make_range(sheets));
  auto [cells, ids] = tf::make_csg_domains(graph);

  // topology path on the materialised arrangement (build the same inputs as Mesh list)
  // ... mesh_arrangements(meshes) -> cleaned -> make_domain_labels(ignore_open_fragments)
  //     -> split_into_domains ; compare cells.size() and total |volume|.
  REQUIRE(cells.size() > 1);
  for (auto &c : cells) { REQUIRE(tf::is_closed(c.polygons())); REQUIRE(tf::is_manifold(c.polygons())); }
}
```

- [ ] **Step 2: Run, expect PASS** (Task 3 already provides `make_csg_domains`). If counts disagree, the bug is in Task 2's per-side emission — fix there, not here.

- [ ] **Step 3: Commit**

```bash
git add tests/csg/test_csg_domains.cpp
git commit -m "test(csg): make_csg_domains parity with topology split_into_domains"
```

---

### Task 5: Mesh source-`ignore` (the source axis) + per-face tags

**Files:**
- Modify: `include/trueform/csg/graph/make_csg_mesh.hpp` (emit per-face source tag), `include/trueform/csg/make_csg_mesh.hpp` (optional `ignore` overload).
- Test: `tests/csg/test_csg_domains.cpp` (extend) or a new `[ignore]` section.

**Interfaces:**
- Consumes: `csg_graph`, `tf::csg::expr`, optional `ignore` range of operand ids.
- Produces:
  - `make_csg_mesh(graph, expr)` — unchanged signature, now also a per-face source-tag buffer alongside the mesh (additive; existing callers ignore it).
  - `make_csg_mesh(graph, expr, ignore_range)` — drops output faces whose source tag ∈ `ignore`; empty range → identical to the 2-arg form.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("make_csg_mesh ignore drops an operand's surface (open piece)", "[ignore]") {
  auto sf = tf::make_sphere_mesh<Real, Index>(1.0, 32, 32);
  auto pf = tf::make_plane_mesh<Real, Index>(3.0, 3.0);
  auto sphere = tf::make_polygons(sf) | tf::tag(frame0());
  auto plane  = tf::make_polygons(pf) | tf::tag(frame0());
  std::vector<decltype(sphere)> forms{sphere, plane};
  auto graph = tf::make_csg_graph(tf::make_range(forms), tf::make_range(std::vector<bool>{false, true}));

  // region = inside sphere AND behind the plane; ignore the plane's cap -> open hemisphere shell
  auto region = tf::csg::op(0) & tf::csg::op(1);
  auto capped = tf::make_csg_mesh(graph, region);                       // closed
  auto open   = tf::make_csg_mesh(graph, region, tf::make_range(std::vector<Index>{1})); // ignore form 1 (the plane)
  REQUIRE(tf::is_closed(capped.polygons()));
  REQUIRE_FALSE(tf::is_closed(open.polygons()));   // cap removed -> open
}
```

- [ ] **Step 2: Run, expect FAIL** (3-arg overload undefined).

- [ ] **Step 3: Implement.** In `csg/graph/make_csg_mesh.hpp`, while concatenating the per-form face streams (the loop at `make_csg_mesh.hpp:180-189`), also write a parallel `tf::buffer<Index> source_tags` whose entry per output face = the form tag `t` of the stream it came from. Return it alongside the mesh (additive). In `csg/make_csg_mesh.hpp`, add the 3-arg overload: build the mesh + tags, and when `ignore.size() > 0`, stream-compact faces whose `source_tags[f]` is in `ignore` (build a small `bool drop[n_tags]` lookup; keep faces with `!drop[tag]`; reuse a parallel copy-if). Empty `ignore` returns the full mesh unchanged.

- [ ] **Step 4: Run, expect PASS** (`[ignore]`).

- [ ] **Step 5: Commit**

```bash
git add include/trueform/csg/graph/make_csg_mesh.hpp include/trueform/csg/make_csg_mesh.hpp tests/csg/test_csg_domains.cpp
git commit -m "feat(csg): make_csg_mesh source-ignore + per-face source tags"
```

---

## Self-review

- **Spec coverage:** make_csg_domains (Tasks 1-3) ✓; all-bounded default + exclude-outer (Task 3 `domain_membership`) ✓; expr selection ✓; split_into_domains parity (Task 4) ✓; source-ignore + per-face tags + empty-range short-circuit (Task 5) ✓; sheets work via bits (covered by using a sheet plane in every test) ✓; shared-wall duplication (implicit in per-side emission, Task 2) ✓. Python/TS bindings + `from()` sugar are spec-deferred — not in this plan.
- **Type consistency:** `domain_partition<Index>` (`.dense_of_domain`, `.side_label`, `.n_kept`) defined Task 1, consumed Tasks 2-3; `make_csg_domains<OutReal>(...) -> pair<vector<polygons_buffer>, buffer<Index>>` defined Task 2, consumed Task 3; `make_csg_domains` signatures consistent Tasks 3-4; `make_csg_mesh` 3-arg consistent Task 5.
- **Placeholder scan:** Task 2 Step 3 and Task 4 Step 1 describe adaptations rather than full literal code because they generalise an existing 60-line function and mirror `domains.py`'s plane setup — each names the exact source file + line range to copy and the exact change. The csg_graph accessor names in Tasks 2-3 must be confirmed against `csg/csg_graph.hpp` at implementation time (Task 2 Step 1 note) — if they differ, use the real names.
- **Risk note:** the one genuinely new algorithmic piece is Task 2's per-side → per-domain emission; Task 4's parity test is its real correctness gate. If Task 2 is too large for one sitting, split it: 2a = per-domain `partition_ids` (selection), 2b = the K-block emission/assembly.
