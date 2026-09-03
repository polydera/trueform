# Arrangement Index Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let clients recover how the output of `make_polygon_arrangements` / `make_mesh_arrangements` relates to the input, via a `tf::return_index_map` overload that returns one finalized struct per function carrying **output-indexed inverse labels** (point/face → input) plus a **forward map** (input point → output point).

**Architecture:** Both functions already build the full point↔input and face↔input mapping internally (`embed_map_data` single-mesh, `arrangement_map_data` N-mesh) and then **discard it**. We add two plain-struct view types that move those buffers in, materialize the point inverse output-indexed (a cheap parallel scatter — faces are already output-indexed), and bake per-mesh offsets into the forward map. New `tf::return_index_map_t` overloads surface them; a shared packager removes per-overload duplication.

**Tech Stack:** Header-only C++17, `tf::buffer`/`tf::small_vector`, Catch2 tests (`tests/cut/`), CMake (`build-tests`).

---

## Final design (settled — do not re-derive)

Two plain structs, all buffers **output-indexed** except `point_f` (input-indexed forward). No `index_map_buffer`, no accessor wrappers — plain members.

```cpp
namespace tf {

// single mesh: no tag axis
template <typename Index> struct polygon_arrangement_index_map {
  tf::buffer<Index> point_labels;   // out point → in point id;  created ⇒ n_output_points
  tf::buffer<Index> face_labels;    // out face  → in face id    (cut faces keep origin)
  tf::buffer<Index> point_f;        // point_f[in id] → out point id   (original → new)
  Index n_original_points = 0;      // out >= this ⇒ created point
  Index n_original_faces  = 0;
  Index n_output_points   = 0;      // the `end` sentinel value
};

// N meshes: add the mesh-tag axis
template <typename Index> struct mesh_arrangement_index_map {
  tf::buffer<Index> point_tag_labels;  // out point → mesh tag;  created ⇒ n_output_points
  tf::buffer<Index> point_labels;      // out point → in point id (local to its mesh)
  tf::buffer<Index> face_tag_labels;   // out face  → mesh tag
  tf::buffer<Index> face_labels;       // out face  → in face id  (cut faces keep origin)
  tf::small_vector<tf::buffer<Index>, 10> point_f;  // point_f[tag][in id] → out point id
  Index n_original_points = 0;
  Index n_original_faces  = 0;
  Index n_output_points   = 0;
};

} // namespace tf
```

**Conventions (identical to the codebase's `index_map` idiom):**
- **`end` sentinel = `n_output_points`** (= `output_mesh.points().size()`), one past every real output index — the index-map analog of `end()`. Created points: `point_tag_labels[o] == n_output_points` (N-mesh) / `point_labels[o] == n_output_points` (single). `point_f` emits the same sentinel for any input id with no output (never happens in the take-all arrangement path, but the convention is uniform with the filtered paths).
- **created test:** `o >= n_original_points` ⇔ created. Created point's id in the new-points space is positional: `o - n_original_points`. Never stored.
- **faces have no `end` tail:** every output face (uncut or cut) keeps a real `(tag, origin-face)`, because a cut face is a piece of an input face. So `face_labels` / `face_tag_labels` are exactly today's `face_labels` / `tag_labels`, unchanged — just moved into the struct.

**Field provenance — everything is already computed, we stop discarding it:**

| struct field | single (`embed_map_data`) | N-mesh (`arrangement_map_data` + its base) |
|---|---|---|
| `point_f` | move `original_map` (already global) | per mesh `t`: `original_map[point_offsets[t]+id] + original_offsets[t]` (= `map_original_vertex`), sliced into `point_f[t]` |
| `point_labels` | `[0,n_orig)=original_ids`, tail `=n_output_points` | scatter: `out=original_offsets[t]+k → original_ids[t][k]`; tail `=n_output_points` |
| `point_tag_labels` | — | scatter: `out=original_offsets[t]+k → t`; tail `=n_output_points` |
| `face_labels` | move construct's `face_labels` | move construct's `face_labels` |
| `face_tag_labels` | — | move construct's `tag_labels` |
| `n_original_points` | `n_original_points` | `total_original_points` |
| `n_original_faces` | `uncut_face_ids.size()` | `total_original_faces` |
| `n_output_points` | passed in (output mesh point count) | `total_original_points + ig.points().size()` |

`arrangement_map_data` fields confirmed present in `include/trueform/cut/construct/arrangement_map_data.hpp`: `original_ids` (small_vector per mesh), `original_map`, `original_offsets`, `point_offsets`, `n_meshes`, `total_original_points`, `total_original_faces`, `map_original_vertex`. `embed_map_data`: `original_ids`, `original_map`, `uncut_face_ids`, `n_original_points`.

**Open item (confirm at execution):** factory name `s` — `make_polygon_arrangement_index_map` vs `make_polygon_arrangements_index_map`. Plan uses the singular form (`..._arrangement_index_map`), matching the type names; function family is plural (`make_polygon_arrangements`).

---

## File Structure

**New:**
- `include/trueform/cut/arrangement_index_map.hpp` — both structs + the two factory functions (consume the respective `*_map_data` by rvalue + `n_output_points`, finalize).
- `tests/cut/test_arrangement_index_map.cpp` — round-trip + structural tests.

**Modified:**
- `include/trueform/cut/make_polygon_arrangements.hpp` — add `return_index_map_t` overloads; hoist the conditional-converter packaging into a file-local helper.
- `include/trueform/cut/make_mesh_arrangements.hpp` — same, for the 2-mesh and N-mesh entry points.
- `include/trueform/cut/construct/make_mesh_arrangements.hpp` — pass `n_output_points` (it computes `total_original_points + ig.points().size()` already at line ~112) to the factory; no return-shape change to the existing tuple.
- `include/trueform/cut.hpp` — `#include "./cut/arrangement_index_map.hpp"` and `#include "./reindex/return_index_map.hpp"`.
- `tests/cut/CMakeLists.txt` — add the new test file.

---

## Task 1: `polygon_arrangement_index_map` (single mesh)

**Files:**
- Create: `include/trueform/cut/arrangement_index_map.hpp`
- Test: `tests/cut/test_arrangement_index_map.cpp`

- [ ] **Step 1: Write the failing test (single-mesh round trip).** In `tests/cut/test_arrangement_index_map.cpp`:

```cpp
#include <trueform/trueform.hpp>
#include <trueform/cut.hpp>
#include <catch2/catch_test_macros.hpp>

// Two triangles sharing an edge, with a third crossing them so the arrangement
// adds intersection points. Reuse the fixture style already in tests/cut.
static auto self_crossing_mesh() {
  tf::polygons_buffer<int, float, 3, 3> m;
  auto &pts = m.points_buffer();
  pts.emplace_back(0.f, 0.f, 0.f); pts.emplace_back(2.f, 0.f, 0.f);
  pts.emplace_back(0.f, 2.f, 0.f); pts.emplace_back(2.f, 2.f, 0.f);
  pts.emplace_back(0.f, 1.f, 0.f); pts.emplace_back(2.f, 1.f, 0.f); // crossing edge ends
  m.faces_buffer().emplace_back(0, 1, 2);
  m.faces_buffer().emplace_back(1, 3, 2);
  m.faces_buffer().emplace_back(4, 5, 3); // overlaps faces 0/1 → creates points
  return m;
}

TEST_CASE("polygon_arrangement_index_map: inverse + forward + created", "[arrangement_index_map]") {
  auto m = self_crossing_mesh();
  auto [out, imap] = tf::make_polygon_arrangements(m.polygons(), tf::return_index_map);

  REQUIRE(imap.n_output_points == int(out.polygons().points().size()));
  // kept originals: inverse points to a valid input id, forward round-trips.
  for (int o = 0; o < imap.n_original_points; ++o) {
    int in = imap.point_labels[o];
    REQUIRE(in >= 0);
    REQUIRE(in < int(m.polygons().points().size()));
    REQUIRE(imap.point_f[in] == o);              // forward original → new
  }
  // created tail: inverse is the end sentinel.
  for (int o = imap.n_original_points; o < imap.n_output_points; ++o)
    REQUIRE(imap.point_labels[o] == imap.n_output_points);
  // every output face carries a valid origin face (no end tail for faces).
  for (std::size_t o = 0; o < out.polygons().size(); ++o) {
    REQUIRE(imap.face_labels[o] >= 0);
    REQUIRE(imap.face_labels[o] < int(m.polygons().size()));
  }
}
```

- [ ] **Step 2: Run it, confirm compile failure** (type + overload missing).

Run: `cmake --build build-tests --target trueform_cut_tests` — Expected: "no matching function for call to make_polygon_arrangements(..., return_index_map_t)".

- [ ] **Step 3: Create the header with the struct + factory.** `include/trueform/cut/arrangement_index_map.hpp`:

```cpp
#pragma once
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "./construct/arrangement_map_data.hpp"

namespace tf {

template <typename Index> struct polygon_arrangement_index_map {
  tf::buffer<Index> point_labels;
  tf::buffer<Index> face_labels;
  tf::buffer<Index> point_f;
  Index n_original_points = 0;
  Index n_original_faces = 0;
  Index n_output_points = 0;
};

/// Finalize an embed_map_data + the already-built face_labels into the public
/// single-mesh type. original_map is already global → point_f is a pure move.
template <typename Index>
auto make_polygon_arrangement_index_map(tf::cut::embed_map_data<Index> &&d,
                                        tf::buffer<Index> &&face_labels,
                                        Index n_output_points)
    -> polygon_arrangement_index_map<Index> {
  polygon_arrangement_index_map<Index> out;
  out.n_original_points = d.n_original_points;
  out.n_original_faces = static_cast<Index>(d.uncut_face_ids.size());
  out.n_output_points = n_output_points;
  out.point_f = std::move(d.original_map);              // in id → out (global)
  out.face_labels = std::move(face_labels);

  out.point_labels.allocate(static_cast<std::size_t>(n_output_points));
  tf::parallel_fill(out.point_labels, n_output_points); // tail = end sentinel
  tf::parallel_for_each(
      tf::make_sequence_range(Index(0), d.n_original_points),
      [&](Index o) { out.point_labels[o] = d.original_ids[o]; });
  return out;
}

} // namespace tf
```

- [ ] **Step 4: Wire the minimal single-mesh `return_index_map` overload.** In `include/trueform/cut/make_polygon_arrangements.hpp`, add an overload that runs the same pipeline but keeps `map_data` and the `face_labels`, and returns `std::make_tuple(mesh, make_polygon_arrangement_index_map(std::move(map_data), std::move(face_labels), Index(mesh.polygons().points().size())))`. (Full wiring + de-dup is Task 3; this is the minimum to compile the test.)

- [ ] **Step 5: Build + run, confirm pass.**

Run: `cmake --build build-tests --target trueform_cut_tests && build-tests/tests/cut/trueform_cut_tests "[arrangement_index_map]"` — Expected: PASS.

- [ ] **Step 6: Commit.** `feat(cut): polygon_arrangement_index_map + return_index_map overload`

---

## Task 2: `mesh_arrangement_index_map` (N-mesh, tags + per-mesh forward)

**Files:**
- Modify: `include/trueform/cut/arrangement_index_map.hpp`
- Modify: `include/trueform/cut/construct/make_mesh_arrangements.hpp` (thread `n_output_points` + keep `tag_labels`/`face_labels` to the factory)
- Test: `tests/cut/test_arrangement_index_map.cpp`

- [ ] **Step 1: Write the failing test (N-mesh per-mesh maps + tags).**

```cpp
TEST_CASE("mesh_arrangement_index_map: tags, per-mesh forward, created", "[arrangement_index_map]") {
  // two overlapping unit boxes offset along x (reuse tests/cut box fixture)
  auto a = /* box at origin, float */;
  auto b = /* box translated (0.5,0,0), float */;
  auto [out, imap] =
      tf::make_mesh_arrangements(a.polygons(), b.polygons(), tf::return_index_map);

  REQUIRE(imap.n_output_points == int(out.polygons().points().size()));
  REQUIRE(int(imap.point_f.size()) == 2);

  for (int o = 0; o < imap.n_original_points; ++o) {
    int t = imap.point_tag_labels[o];
    REQUIRE((t == 0 || t == 1));
    int in = imap.point_labels[o];
    REQUIRE(in >= 0);
    REQUIRE(imap.point_f[t][in] == o);            // per-mesh forward round-trips
  }
  for (int o = imap.n_original_points; o < imap.n_output_points; ++o) {
    REQUIRE(imap.point_tag_labels[o] == imap.n_output_points);
    REQUIRE(imap.point_labels[o] == imap.n_output_points);
  }
  for (std::size_t o = 0; o < out.polygons().size(); ++o) {
    REQUIRE((imap.face_tag_labels[o] == 0 || imap.face_tag_labels[o] == 1));
    REQUIRE(imap.face_labels[o] >= 0);
  }
}
```

- [ ] **Step 2: Run it, confirm compile failure.**

- [ ] **Step 3: Add the N-mesh struct + factory** to `arrangement_index_map.hpp`:

```cpp
namespace tf {

template <typename Index> struct mesh_arrangement_index_map {
  tf::buffer<Index> point_tag_labels;
  tf::buffer<Index> point_labels;
  tf::buffer<Index> face_tag_labels;
  tf::buffer<Index> face_labels;
  tf::small_vector<tf::buffer<Index>, 10> point_f;
  Index n_original_points = 0;
  Index n_original_faces = 0;
  Index n_output_points = 0;
};

/// Finalize arrangement_map_data + the already-built tag_labels/face_labels.
/// point inverse is scattered output-indexed; point_f is sliced per mesh with
/// original_offsets baked so each entry is a global output index.
template <typename Index>
auto make_mesh_arrangement_index_map(tf::cut::arrangement_map_data<Index> &&d,
                                     tf::buffer<Index> &&tag_labels,
                                     tf::buffer<Index> &&face_labels,
                                     Index n_output_points)
    -> mesh_arrangement_index_map<Index> {
  mesh_arrangement_index_map<Index> out;
  const Index n = d.n_meshes;
  out.n_original_points = d.total_original_points;
  out.n_original_faces = d.total_original_faces;
  out.n_output_points = n_output_points;
  out.face_tag_labels = std::move(tag_labels);
  out.face_labels = std::move(face_labels);

  // point inverse, output-indexed, with end-sentinel tail.
  out.point_tag_labels.allocate(static_cast<std::size_t>(n_output_points));
  out.point_labels.allocate(static_cast<std::size_t>(n_output_points));
  tf::parallel_fill(out.point_tag_labels, n_output_points);
  tf::parallel_fill(out.point_labels, n_output_points);
  for (Index t = 0; t < n; ++t) {
    const Index base = d.original_offsets[t];
    const auto &ids = d.original_ids[t];
    tf::parallel_for_each(
        tf::make_sequence_range(Index(0), Index(ids.size())), [&](Index k) {
          const Index o = base + k;
          out.point_tag_labels[o] = t;
          out.point_labels[o] = ids[k];
        });
  }

  // forward, per mesh, offsets baked → global output index.
  out.point_f.resize(n);
  for (Index t = 0; t < n; ++t) {
    const Index lo = d.point_offsets[t];
    const Index cnt = d.point_offsets[t + 1] - lo;
    const Index off = d.original_offsets[t];
    tf::buffer<Index> f;
    f.allocate(static_cast<std::size_t>(cnt));
    tf::parallel_for_each(tf::make_sequence_range(Index(0), cnt), [&](Index v) {
      f[v] = d.original_map[lo + v] + off;
    });
    out.point_f[t] = std::move(f);
  }
  return out;
}

} // namespace tf
```

- [ ] **Step 4: Thread `n_output_points` + labels out of the construct.** In `include/trueform/cut/construct/make_mesh_arrangements.hpp` the tuple already returns `(mesh, tag_labels, face_labels, map_data)` and computes `total_original_points + ig.points().size()` (~line 112). No change to that tuple. The public overload (Task 3) computes `n_output_points = int(mesh.polygons().points().size())` from the returned mesh and feeds `make_mesh_arrangement_index_map(std::move(map_data), std::move(tag_labels), std::move(face_labels), n_output_points)`.

- [ ] **Step 5: Wire the minimal 2-mesh `return_index_map` overload** in `make_mesh_arrangements.hpp` (full wiring Task 3). Build + run.

Run: `cmake --build build-tests --target trueform_cut_tests && build-tests/tests/cut/trueform_cut_tests "[arrangement_index_map]"` — Expected: PASS.

- [ ] **Step 6: Commit.** `feat(cut): mesh_arrangement_index_map (tags + per-mesh point_f)`

---

## Task 3: `return_index_map` overloads + de-duplicate the converter packaging

**Files:**
- Modify: `include/trueform/cut/make_polygon_arrangements.hpp`
- Modify: `include/trueform/cut/make_mesh_arrangements.hpp`
- Modify: `include/trueform/cut.hpp`

The risk: the `if constexpr (!is_integral<InputReal> && is_integral<RealOut>)` converter-append branch being copy-pasted into every new overload. Hoist it into one file-local helper so each variant is a thin call.

- [ ] **Step 1: Add includes + umbrella export.** In both public headers: `#include "../reindex/return_index_map.hpp"` and `#include "./arrangement_index_map.hpp"`. In `cut.hpp`: add both.

- [ ] **Step 2: Factor the conditional-converter append into one file-local `pack(...)` helper** in each public header (no public API). It takes `(mesh, extras..., converter)` and appends `converter` to the returned tuple only in the float-in/int-out case via `if constexpr`. Both the existing base overloads and the new `return_index_map` overloads route through it, so the branch lives once.

- [ ] **Step 3: Add `make_polygon_arrangements` `return_index_map` overloads** (config+tag, and default-config+tag, mirroring the `return_curves` overloads). Keep `map_data` + `face_labels`; return `pack(mesh, make_polygon_arrangement_index_map(std::move(map_data), std::move(face_labels), Index(mesh.polygons().points().size())), converter)`.

```cpp
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &p,
                               tf::intersect_config config,
                               tf::return_index_map_t) { /* keep map_data; pack(...) */ }

template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &p,
                               tf::return_index_map_t) {
  return make_polygon_arrangements<Int, OutputCoordinateType>(
      p, {tf::intersect_mode::primitives | tf::intersect_mode::resolve_contours},
      tf::return_index_map);
}
```

- [ ] **Step 4: Add `make_mesh_arrangements` `return_index_map` overloads** for both the 2-mesh (`p0,p1[,config]`) and N-mesh range (`forms[,config]`) entries. Each keeps `tag_labels`/`face_labels`/`map_data` and returns `pack(mesh, make_mesh_arrangement_index_map(std::move(map_data), std::move(tag_labels), std::move(face_labels), Index(mesh.polygons().points().size())), converter)`.

- [ ] **Step 5: Build the whole cut test target + run.**

Run: `cmake --build build-tests --target trueform_cut_tests && build-tests/tests/cut/trueform_cut_tests "[arrangement_index_map]"` — Expected: PASS.

- [ ] **Step 6: Commit.** `feat(cut): return_index_map overloads for make_*_arrangements`

---

## Task 4: float→int converter path coverage

**Files:**
- Test: `tests/cut/test_arrangement_index_map.cpp`

The conditional return appends a `converter` only for float-input / integer-output. The index map must sit in the right tuple slot in that case too.

- [ ] **Step 1: Add a test with `OutputCoordinateType = std::int32_t` (float input)** for both polygon and mesh paths; structured-bind `[out, imap, conv]` and assert the same invariants as Tasks 1–2, plus `conv.deconvert(...)` on an output point is finite/sane.

- [ ] **Step 2: Run, confirm pass** (or fix the `pack` tuple order if the converter lands in the wrong slot).

- [ ] **Step 3: Commit.** `test(cut): arrangement index map under float->int output`

---

## Task 5: docs + umbrella sanity

**Files:**
- Modify: `include/trueform/cut/arrangement_index_map.hpp` (doc comments)
- Modify: `CLAUDE.md` arrangement section (optional one-liner)

- [ ] **Step 1: Doxygen `@ingroup cut_boolean` + brief** documenting the `end` sentinel (`n_output_points`), the created test (`o >= n_original_points`), and that `point_f` is the only forward map (faces need none).
- [ ] **Step 2: Compile-only test** that includes just `<trueform/cut.hpp>` and references `tf::return_index_map`, both structs, and a `return_index_map` overload.
- [ ] **Step 3: Commit.** `docs(cut): document arrangement index maps`

---

## Self-Review (run before handing off)

1. **Spec coverage:** two structs (Tasks 1,2); output-indexed point inverse via scatter + end tail (Task 2 Step 3); per-mesh forward with baked offsets (Task 2 Step 3); faces moved verbatim from existing `tag_labels`/`face_labels` (Tasks 1–2); `return_index_map` overloads on both functions (Task 3); single `pack` helper, no duplicated converter branch (Task 3 Step 2); float→int slot correctness (Task 4).
2. **Placeholder scan:** the two test fixtures ("self_crossing_mesh", "two overlapping boxes") are concrete in Task 1 and must be made concrete in Task 2 Step 1 before running (reuse an existing `tests/cut` box/overlap fixture).
3. **Type consistency:** members `point_labels`, `face_labels`, `point_tag_labels`, `face_tag_labels`, `point_f`, `n_original_points`, `n_original_faces`, `n_output_points`; factories `make_polygon_arrangement_index_map` / `make_mesh_arrangement_index_map` — used identically across header, factories, and tests. `end` sentinel is `n_output_points` everywhere.
