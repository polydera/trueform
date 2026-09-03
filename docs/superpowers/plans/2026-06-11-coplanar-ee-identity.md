# Coplanar EE-crossing identity fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the EE-crossing identity from over-merging distinct crossings when faces are coplanar, without splitting genuine 3-plane junctions.

**Architecture:** EE crossing points are currently keyed solely on the face-triple `{A,B,D}` (`compute_ee_crossing_points` groups by `a.triple == b.triple`). That assumes a triple meets at one point — false when a face-pair is coplanar and meets in ≥2 contact edges, so a third edge crosses both and the two distinct crossings collapse to one. The fix propagates the *exact* coplanarity bit that the classifier already decides at edge construction (`edge_extractor::extract_coplanar`) onto each edge, carries it onto each crossing record, and refines the identity key to `(triple, coplanar ? (edge_a,edge_b) : (0,0))`. Generic junctions (both edges transversal) still merge by triple — robust, unchanged. Coplanar-pack crossings refine by canonical-edge pair — and that is safe precisely because the two coplanar faces share a plane, so both report the same edge-pair (no cross-face under-merge). No geometry is re-derived from rounded coordinates.

**Tech Stack:** C++17 header-only (trueform), Catch2 tests, CMake.

---

## Background (root cause, confirmed)

- Identity key today: `crossing_points.hpp:51` `compute_offsets(... a.triple == b.triple)`; sorted by `crossing_classification.hpp:36` `a.triple < b.triple`. One point per triple.
- `crossing_split_entries.hpp:45` assigns every record in a triple-group the same `pid = crossing_base + g`, coordinate from `group[0]` only → the other crossings are lost / mis-split.
- Reproduced: Lévy `nasty_gears` (rotated cubes sharing Z-planes), N=4 → 2 over-merged packs, N=5 → 1; **zero** when de-coplanarized (`TF_ZJIT`). Diagnostic lived behind `#ifdef TF_DUMP_EE_TRIPLE` in `crossing_points.hpp` (Task 6 removes it).
- Why the naive "edge-pair only" key (tried before) regressed: a generic junction `Q` sits on 3 edges and is detected as 3 different edge-pairs across its 3 faces; an edge-pair key splits `Q` into up to 3 ids that only re-merge if their `div_round` coordinates land in the same cell (they need not). The triple merges them robustly. So the key must stay triple **except** inside coplanar packs — distinguished by the propagated coplanar bit, not geometry.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `include/trueform/intersect/graph/edge.hpp` | edge instance struct | add `bool from_coplanar = false;` |
| `include/trueform/intersect/graph/crossing_record.hpp` | crossing record struct | add `bool coplanar = false;` |
| `include/trueform/intersect/graph/edge_extractor.hpp` | edge extraction | stamp `from_coplanar=true` on edges from `extract_coplanar` |
| `include/trueform/intersect/graph/crossing_detection.hpp` | EE/VE/VV record emit | set record `coplanar` from `ea/eb.from_coplanar` |
| `include/trueform/intersect/graph/crossing_classification.hpp` | record sort | refine EE comparator key |
| `include/trueform/intersect/graph/crossing_points.hpp` | EE point grouping | refine grouping predicate; remove temp diagnostic |
| `tests/intersect/test_intersection_graph.cpp` | tests | add 2 unit tests on the grouping key |

---

### Task 1: Add the data fields (no behavior change yet)

**Files:**
- Modify: `include/trueform/intersect/graph/edge.hpp:19-29`
- Modify: `include/trueform/intersect/graph/crossing_record.hpp:35-42`

- [ ] **Step 1: Add `from_coplanar` to `edge`**

In `edge.hpp`, the struct becomes:

```cpp
template <typename Index> struct edge {
  short tag;
  short tag_other;
  Index object;
  Index object_other;
  Index point_0;
  Index point_1;
  Index id; // canonical group ID
  std::int16_t ordinal;     // base-loop position of start vertex; -1 if interior
  std::int16_t sub_ordinal; // parametric order along parent base-loop segment; -1 if interior
  bool from_coplanar = false; // emitted from a coplanar polygon-of-contact
};
```

(The 9-field positional aggregate inits in `edges.hpp` stay valid; `from_coplanar` defaults to false.)

- [ ] **Step 2: Add `coplanar` to `crossing_record`**

In `crossing_record.hpp`:

```cpp
template <typename Index> struct crossing_record {
  Index edge_a;
  Index edge_b;
  Index point_a;
  Index point_b;
  std::array<face_id<Index>, 3> triple;
  enum type_t : uint8_t { ee, ve, vv } type;
  bool coplanar = false; // EE crossing involves a coplanar-contact edge
};
```

- [ ] **Step 3: Build the existing intersect tests to confirm nothing broke**

Run: `cmake --build build --target trueform_intersect_tests`
Expected: builds clean (fields are additive with defaults).

- [ ] **Step 4: Commit**

```bash
git add include/trueform/intersect/graph/edge.hpp include/trueform/intersect/graph/crossing_record.hpp
git commit -m "intersect: add coplanar flags to edge and crossing_record"
```

---

### Task 2: Write the failing unit tests (key behavior)

**Files:**
- Test: `tests/intersect/test_intersection_graph.cpp` (append)

These tests call the real grouping path (`sort_crossing_records` + `compute_ee_crossing_points`) on hand-built records — no meshes, no rounding, answers known a priori.

- [ ] **Step 1: Add includes (top of file, after existing includes)**

```cpp
#include <trueform/intersect/graph/crossing_classification.hpp>
#include <trueform/intersect/graph/crossing_points.hpp>
```

- [ ] **Step 2: Append the two tests**

```cpp
// =============================================================================
// EE crossing identity: coplanar packs must NOT over-merge; generic junctions
// must still merge by triple. (Regression: coplanar EE over-merge.)
// =============================================================================
namespace {
using rec_t = tf::intersect::graph::crossing_record<int>;
using fid = tf::intersect::graph::face_id<int>;

// Build edge_defs holding three canonical edges, each a 1-element group:
//   edge 0: p0->p1, edge 1: p2->p3, edge 2: p4->p5.
auto make_edge_defs(const std::array<std::array<int, 2>, 3> &eps)
    -> tf::offset_block_buffer<int, tf::intersect::graph::edge<int>> {
  tf::offset_block_buffer<int, tf::intersect::graph::edge<int>> defs;
  for (int i = 0; i < 3; ++i) {
    tf::intersect::graph::edge<int> e{};
    e.point_0 = eps[i][0];
    e.point_1 = eps[i][1];
    e.id = i;
    defs.push_back({e});
  }
  return defs;
}

// Count distinct EE crossing points produced by the real grouping path.
auto count_ee_points(std::vector<rec_t> records,
                     const tf::offset_block_buffer<int, tf::intersect::graph::edge<int>> &defs,
                     const tf::points_buffer<int, 3> &pts) -> std::size_t {
  tf::buffer<rec_t> buf;
  for (auto &r : records)
    buf.push_back(r);
  tf::intersect::graph::sort_crossing_records<int>(buf);
  auto [ee, ve, vv] = tf::intersect::graph::find_type_ranges<int>(buf);
  auto get_point = [&](int, int id) { return pts[id]; };
  tf::buffer<tf::point<int, 3>> crossing_points;
  auto offs = tf::intersect::graph::compute_ee_crossing_points<int>(
      ee, defs, get_point, crossing_points);
  return offs.size() > 0 ? offs.size() - 1 : 0;
}
} // namespace

TEST_CASE("coplanar pack: two crossings on one triple stay distinct",
          "[intersection_graph][coplanar]") {
  // Plane z=0. Edge 2 (L) is the x-axis [0,0]..[10,0]; edge 0 is vertical at
  // x=2, edge 1 vertical at x=6. L crosses edge0 at (2,0) and edge1 at (6,0):
  // two distinct points, same triple {A,B,D}, both coplanar-involved.
  tf::points_buffer<int, 3> pts;
  pts.emplace_back(2, -1, 0);  // 0  edge0
  pts.emplace_back(2, 1, 0);   // 1
  pts.emplace_back(6, -1, 0);  // 2  edge1
  pts.emplace_back(6, 1, 0);   // 3
  pts.emplace_back(0, 0, 0);   // 4  edge2 (L)
  pts.emplace_back(10, 0, 0);  // 5
  auto defs = make_edge_defs({{{0, 1}, {2, 3}, {4, 5}}});

  std::array<fid, 3> triple = {fid{0, 0}, fid{1, 0}, fid{2, 0}};
  rec_t r0{};
  r0.edge_a = 0; r0.edge_b = 2; r0.triple = triple; r0.type = rec_t::ee; r0.coplanar = true;
  rec_t r1{};
  r1.edge_a = 1; r1.edge_b = 2; r1.triple = triple; r1.type = rec_t::ee; r1.coplanar = true;

  REQUIRE(count_ee_points({r0, r1}, defs, pts) == 2);
}

TEST_CASE("generic junction: three edge-pairs on one triple still merge",
          "[intersection_graph][coplanar]") {
  // Three edges concurrent at the origin in z=0; the junction is detected as
  // three edge-pairs (0,1),(0,2),(1,2), all NOT coplanar -> must collapse to 1.
  tf::points_buffer<int, 3> pts;
  pts.emplace_back(-5, 0, 0);  // 0  edge0 horizontal through origin
  pts.emplace_back(5, 0, 0);   // 1
  pts.emplace_back(0, -5, 0);  // 2  edge1 vertical through origin
  pts.emplace_back(0, 5, 0);   // 3
  pts.emplace_back(-5, -5, 0); // 4  edge2 diagonal through origin
  pts.emplace_back(5, 5, 0);   // 5
  auto defs = make_edge_defs({{{0, 1}, {2, 3}, {4, 5}}});

  std::array<fid, 3> triple = {fid{0, 0}, fid{1, 0}, fid{2, 0}};
  auto mk = [&](int a, int b) {
    rec_t r{};
    r.edge_a = a; r.edge_b = b; r.triple = triple; r.type = rec_t::ee; r.coplanar = false;
    return r;
  };
  REQUIRE(count_ee_points({mk(0, 1), mk(0, 2), mk(1, 2)}, defs, pts) == 1);
}
```

- [ ] **Step 3: Build and run — the pack test must FAIL, the junction test must PASS**

Run:
```bash
cmake --build build --target trueform_intersect_tests && \
  ./build/tests/intersect/trueform_intersect_tests "[coplanar]" -s
```
Expected: `coplanar pack...` FAILS (gets 1, expected 2 — the bug). `generic junction...` PASSES (1). This proves the test discriminates the bug.

- [ ] **Step 4: Commit the tests (red)**

```bash
git add tests/intersect/test_intersection_graph.cpp
git commit -m "test: EE coplanar-pack over-merge regression (red)"
```

---

### Task 3: Refine the identity key (the fix)

**Files:**
- Modify: `include/trueform/intersect/graph/crossing_classification.hpp:32-40`
- Modify: `include/trueform/intersect/graph/crossing_points.hpp:49-51`

- [ ] **Step 1: Refine the EE sort comparator**

In `sort_crossing_records`, replace the EE branch (`if (a.type == ... ee) return a.triple < b.triple;`) with:

```cpp
        if (a.type == crossing_record<Index>::ee) {
          if (a.coplanar != b.coplanar)
            return a.coplanar < b.coplanar;
          if (a.triple != b.triple)
            return a.triple < b.triple;
          if (a.coplanar)
            return std::tie(a.edge_a, a.edge_b) <
                   std::tie(b.edge_a, b.edge_b);
          return false;
        }
```

- [ ] **Step 2: Refine the EE grouping predicate**

In `compute_ee_crossing_points`, replace the `tf::compute_offsets(... )` predicate `[](const auto &a, const auto &b) { return a.triple == b.triple; }` with:

```cpp
      [](const auto &a, const auto &b) {
        if (a.coplanar != b.coplanar)
          return false;
        if (a.triple != b.triple)
          return false;
        if (a.coplanar)
          return a.edge_a == b.edge_a && a.edge_b == b.edge_b;
        return true;
      });
```

- [ ] **Step 3: Run the unit tests — both must PASS now**

Run:
```bash
cmake --build build --target trueform_intersect_tests && \
  ./build/tests/intersect/trueform_intersect_tests "[coplanar]" -s
```
Expected: both PASS (pack → 2, junction → 1).

- [ ] **Step 4: Run the full intersect + cut suites — no regressions**

Run:
```bash
ctest --test-dir build -R "intersect::|cut::" --output-on-failure
```
Expected: all pass.

- [ ] **Step 5: Commit (green)**

```bash
git add include/trueform/intersect/graph/crossing_classification.hpp include/trueform/intersect/graph/crossing_points.hpp
git commit -m "intersect: key EE identity by edge-pair inside coplanar packs"
```

---

### Task 4: Wire the real coplanar bit through the pipeline

Until now the unit tests set `coplanar`/`from_coplanar` by hand. This task makes the real arrangement set them so the gear (and all real input) benefits.

**Files:**
- Modify: `include/trueform/intersect/graph/edge_extractor.hpp:78-122`
- Modify: `include/trueform/intersect/graph/crossing_detection.hpp:65-80`

- [ ] **Step 1: Stamp `from_coplanar` on coplanar-extracted edges**

In `extract_coplanar`, record the buffer position on entry and stamp every edge it appended. Change the body so it begins by capturing the start index and ends by stamping:

At the top of `extract_coplanar`, immediately after `_work.clear();`:

```cpp
    auto cp_start = buf.size();
```

Immediately before the closing `}` of `extract_coplanar` (after the `while` loop that emits edges):

```cpp
    for (std::size_t i = cp_start; i < buf.size(); ++i)
      buf[i].from_coplanar = true;
```

- [ ] **Step 2: Set the record `coplanar` flag in the EE emit**

In `crossing_detection.hpp` `test_edge_pair`, the EE branch currently pushes:

```cpp
      out.push_back({lo,
                     hi,
                     {},
                     {},
                     {faces[0], faces[1], faces[2]},
                     crossing_record<Index>::ee});
```

Change it to set `coplanar` from the source edges:

```cpp
      out.push_back({lo,
                     hi,
                     {},
                     {},
                     {faces[0], faces[1], faces[2]},
                     crossing_record<Index>::ee,
                     ea.from_coplanar || eb.from_coplanar});
```

(VE/VV emits are left unchanged; `coplanar` defaults to false there.)

- [ ] **Step 3: Build the cut tests**

Run: `cmake --build build --target trueform_cut_tests trueform_intersect_tests`
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add include/trueform/intersect/graph/edge_extractor.hpp include/trueform/intersect/graph/crossing_detection.hpp
git commit -m "intersect: propagate coplanar-contact bit from extractor to records"
```

---

### Task 5: Integration validation on the gear (the original repro)

**Files:** none modified (validation only). Uses `experimentation/csg_gear_det.cpp` + the temp diagnostic still present in `crossing_points.hpp`.

- [ ] **Step 1: Rebuild the gear diagnostic harness**

Run:
```bash
cd /Users/ziga/trueform/experimentation && \
TBB_INC=$(brew --prefix tbb)/include; TBB_LIB=$(brew --prefix tbb)/lib; \
clang++ -std=c++17 -O2 -DTF_DUMP_EE_TRIPLE -I../include -I"$TBB_INC" \
  csg_gear_det.cpp -L"$TBB_LIB" -ltbb -o /tmp/gear_det_diag
```
Expected: builds clean.

- [ ] **Step 2: Confirm over-merge is gone for the coplanar gear**

Run:
```bash
for N in 3 4 5; do printf "N=%s " $N; \
  TF_THREADS=1 TF_DUMP_EE_TRIPLE=1 /tmp/gear_det_diag $N 2>&1 | grep -oE "OVER_MERGED.*"; done
```
Expected: `OVER_MERGED_coplanar_packs=0  crossings_lost>=0` for **all** N (was 2 for N=4, 1 for N=5 before the fix).

- [ ] **Step 3: Full test suite green**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: all suites pass (no regression in topology/cut/intersect/csg).

---

### Task 6: Remove the temporary diagnostic and finalize

**Files:**
- Modify: `include/trueform/intersect/graph/crossing_points.hpp`

- [ ] **Step 1: Remove the `#ifdef TF_DUMP_EE_TRIPLE` block and its includes**

Delete the `#ifdef TF_DUMP_EE_TRIPLE ... #endif` diagnostic block after the `parallel_for_each` in `compute_ee_crossing_points`, and the matching `#ifdef TF_DUMP_EE_TRIPLE` include block near the top (`<array> <cstdio> <cstdlib> <utility>`).

- [ ] **Step 2: Build and run intersect tests once more**

Run:
```bash
cmake --build build --target trueform_intersect_tests && \
  ./build/tests/intersect/trueform_intersect_tests "[coplanar]" -s
```
Expected: both `[coplanar]` tests pass.

- [ ] **Step 3: Commit**

```bash
git add include/trueform/intersect/graph/crossing_points.hpp
git commit -m "intersect: remove temporary EE-triple over-merge diagnostic"
```

---

## Self-review notes

- **Spec coverage:** root cause (key) → Task 3; data plumbing → Tasks 1,4; reproduction/validation → Tasks 2 (unit), 5 (gear); cleanup → Task 6.
- **Type consistency:** `from_coplanar` (edge), `coplanar` (crossing_record) used consistently; comparator and predicate use the identical key logic `(coplanar, triple, coplanar?(edge_a,edge_b):equal)`.
- **No-regression guard:** the "generic junction" unit test (Task 2) and the full suite (Tasks 3,5) ensure 3-plane junctions still merge by triple.
- **Perf:** one `bool` per edge/record + one extra comparison in the EE comparator/predicate. No new passes, no geometry. Neutral-to-faster (no re-derivation).

## Out of scope
- The rare point that is simultaneously a coplanar-pack crossing **and** a 3-distinct-plane junction (would need a Level-2 exact-position merge). Not observed in the gear corpus; revisit only if a real case surfaces.
- The soup `make_polygon_arrangements` path resolves flat coplanar contacts via the contour resolver and does not exercise this key; unaffected.
