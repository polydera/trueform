# TypeScript Parallel Structure Build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build each mesh's spatial+topology structures (AABB tree, face_membership, manifold_edge_link) concurrently across meshes in the TypeScript/wasm cut + intersect dispatch paths, mirroring the C++ bench's `parallel_invoke(s0, s1)` and the Python binding work already done — so the first boolean/arrangement/intersection on fresh meshes doesn't serialize the structure builds.

**Architecture:** wasm is genuinely multi-threaded (`-pthread -sUSE_PTHREADS=1 -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency'`, TBB linked). Every `mesh_data::ensure_*` builder is pure-C++ compute + a `wasm_*::from_buffer(std::move(...))` that touches **no** `emscripten::val` (proven: the async path already runs these on a worker thread). There is therefore **no GIL-equivalent constraint** — unlike Python we need no compute/commit split. We just trigger the (already cached, build-if-needed) structure builds via `tbb::parallel_invoke` before the dispatch reads them. A small shared header provides per-mesh / pair / N-mesh build helpers; each sync driver calls it once up front; the existing lazy getters then hit the cache. **The helper gates every task dispatch on the `is_*_fresh()` inspectors** (which equal the exact `ensure_*` skip condition) — so we only spawn TBB tasks for structures that actually need building, and a fully-cached mesh/pair/list spawns no parallel work at all (zero overhead on the reused-mesh path).

**Tech Stack:** Emscripten/wasm, oneTBB (`tbb::parallel_invoke` / `tbb::task_group`), C++17, `wasm_mesh<Real>` handles (shared_ptr to `mesh_data<Real>`), esbuild bundle, node test harness.

**Scope (user-specified):** boolean, polygon arrangements, mesh arrangements, the entire intersect module. Embedded intersection/self-intersection (cut module, identical pattern) is included as a clearly-marked task — include or drop per preference. **Out of scope:** isobands / isocontours (they pass only `polygons_range()` + `scalars`, never tree/fm/mel — verified), and the half_edges/face_link/vertex_link/normals structures (not used by these gold paths).

**Reference (already shipped, mirror these):**
- Python: `python/include/trueform/python/intersect/build_intersect_structures.hpp` (the helper shape) and the 11 dispatch call sites.
- C++ core: `include/trueform/cut/dispatch/boolean.hpp` `make_missing_structures` (`tbb::parallel_invoke(tree, {fm;mel})`) and `include/trueform/cut/dispatch/arrangement.hpp` (`tbb::task_group` over forms).

**Tree config:** already `config_tree(4, 12)` in `typescript/cpp/src/core/mesh_data.cpp:34` (matches the C++ bench). No change needed.

---

## Key facts grounding this plan (verified, do not re-derive)

- `wasm_mesh<Real>` (`typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp`) exposes lazy accessors that build-on-first-access and cache via a generation counter; the only explicit void builder today is `build_tree() -> void` (line 107, body `(void)_data->tree();`). There is **no** `build_face_membership()` / `build_manifold_edge_link()` — Task 1 adds them.
- The lazy range getters that trigger the topology builds are `face_membership_range()` (wasm_mesh.hpp:153 → `_data->face_membership_range()` → `ensure_face_membership()`) and `manifold_edge_link_range()` (wasm_mesh.hpp:156 → `ensure_manifold_edge_link()`).
- **Dependency:** `mesh_data::ensure_manifold_edge_link()` (mesh_data.cpp ~105) calls `ensure_face_membership()` first. So within one mesh, fm must precede mel; tree is independent. → per-mesh build = `parallel_invoke(tree, {fm; mel})`.
- **Thread-safety:** every `ensure_*` is pure C++ + `from_buffer` (std::make_shared + move), no `emscripten::val`. The async wrappers already run the sync drivers (hence these builds) on a pthread worker via `promise([...])`. Safe on TBB workers.
- **Caching is already present:** `ensure_*` early-returns when `_X.is_valid() && _X_gen == _faces_gen`. So calling the build helper when structures are fresh is a no-op (build-if-needed). Reused meshes pay nothing.
- **N-mesh entry points** pass `std::vector<wasm_mesh<Real>>` (extracted from a JS array on the main thread by `extract_meshes`), iterated in `mesh_arrangement_impl.hpp` `run_arrangement`/`run_arrangement_with_curves` and `intersect_impl.hpp` `intersection_curves_list_impl`.

---

## File Structure

**New:**
- `typescript/cpp/include/trueform/ts/core/build_intersect_structures.hpp` — per-mesh / pair / N-mesh parallel build helpers. Single shared header, included by cut + intersect dispatch impls.

**Modified:**
- `typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp` — add `build_face_membership()` and `build_manifold_edge_link()` void builders (mirror `build_tree()`).
- `typescript/cpp/src/cut/boolean_impl.hpp` — `sync_boolean`, `sync_boolean_with_curves` (PAIR).
- `typescript/cpp/src/cut/polygon_arrangement_impl.hpp` — `sync_polygon_arrangements`, `sync_polygon_arrangements_with_curves` (SELF).
- `typescript/cpp/src/cut/mesh_arrangement_impl.hpp` — `run_arrangement`, `run_arrangement_with_curves` (N-MESH).
- `typescript/cpp/src/intersect/intersect_impl.hpp` — `sync_intersection_curves` (PAIR), `intersection_curves_list_impl` (N-MESH), `sync_self_intersection_curves` (SELF).
- (optional, Task 7) `typescript/cpp/src/cut/embedded_impl.hpp` — `sync_embedded_intersection_curves`(+`_with_curves`) (PAIR), `sync_embedded_self_intersection_curves`(+`_with_curves`) (SELF).

**Ordering rationale:** Task 1 adds the wasm_mesh builders (no behavior change). Task 2 adds the helper (no call sites yet). Tasks 3–6 wire the in-scope gold paths. Task 7 is the optional embedded paths. Task 8 builds + runs the TS suite.

---

## Task 1: Add `build_face_membership` / `build_manifold_edge_link` to `wasm_mesh`

**Files:**
- Modify: `typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp` (near line 107, beside `build_tree()`)

- [ ] **Step 1: Read the existing `build_tree()` and the lazy range getters** to confirm exact names/bodies.

Run: `sed -n '103,157p' typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp`
Expected: `auto build_tree() -> void { (void)_data->tree(); }` at ~107; `face_membership_range()` at ~153; `manifold_edge_link_range()` at ~156.

- [ ] **Step 2: Add the two void builders** immediately after `build_tree()`:

```cpp
  auto build_face_membership() -> void { (void)_data->face_membership_range(); }
  auto build_manifold_edge_link() -> void {
    (void)_data->manifold_edge_link_range();
  }
```

(These delegate to the same `ensure_*` path the lazy getters use; `manifold_edge_link_range()` internally ensures face_membership first, so calling `build_manifold_edge_link()` alone is also valid — but the helper will call fm then mel explicitly for clarity.)

- [ ] **Step 3: Confirm it compiles** as part of Task 8's build (no standalone wasm unit build here). No comment on these one-liners (match the comment-free `build_tree()`).

- [ ] **Step 4: Commit.**

```bash
git add typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp
git commit -m "feat(ts): wasm_mesh build_face_membership/build_manifold_edge_link void builders"
```

---

## Task 2: The shared parallel-build helper

**Files:**
- Create: `typescript/cpp/include/trueform/ts/core/build_intersect_structures.hpp`

- [ ] **Step 1: Create the header** (full canonical copyright header — copy verbatim from `typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp` lines 1–12). No doc-comments on the helpers (match house style; the TS dispatch code is comment-sparse). Content:

```cpp
/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "trueform/ts/core/wasm_mesh.hpp"
#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>
#include <vector>

namespace tf {
namespace ts {

// `is_*_fresh()` is the EXACT condition ensure_* uses to skip (verified:
// mesh_data.hpp:288/300/305). So `!is_*_fresh()` is a cheap, accurate predicate
// for "this structure would actually build" — we gate every task dispatch on it
// so cached/fresh meshes spawn no TBB work at all.

template <typename Real>
auto needs_intersect_structures(const wasm_mesh<Real> &m) -> bool {
  return !m.is_tree_fresh() || !m.is_face_membership_fresh() ||
         !m.is_manifold_edge_link_fresh();
}

template <typename Real>
auto build_intersect_structures(wasm_mesh<Real> &m) -> void {
  bool need_tree = !m.is_tree_fresh();
  bool need_topo =
      !m.is_face_membership_fresh() || !m.is_manifold_edge_link_fresh();
  auto build_topo = [&] {
    m.build_face_membership();
    m.build_manifold_edge_link();
  };
  if (need_tree && need_topo)
    tbb::parallel_invoke([&] { m.build_tree(); }, build_topo);
  else if (need_tree)
    m.build_tree();
  else if (need_topo)
    build_topo();
}

template <typename Real>
auto build_intersect_structures(wasm_mesh<Real> &a, wasm_mesh<Real> &b)
    -> void {
  if (a.same_data(b)) {
    build_intersect_structures(a);
    return;
  }
  bool na = needs_intersect_structures(a);
  bool nb = needs_intersect_structures(b);
  if (na && nb)
    tbb::parallel_invoke([&] { build_intersect_structures(a); },
                         [&] { build_intersect_structures(b); });
  else if (na)
    build_intersect_structures(a);
  else if (nb)
    build_intersect_structures(b);
}

// Precondition: meshes are distinct (no two entries share one mesh_data). An
// arrangement of N meshes is by definition distinct inputs; we do NOT dedup
// (a pairwise same_data check would be O(n^2)). Duplicate meshes in the list
// are unsupported.
template <typename Real>
auto build_intersect_structures_all(std::vector<wasm_mesh<Real>> &meshes)
    -> void {
  tbb::task_group tg;
  bool any = false;
  for (auto &m : meshes)
    if (needs_intersect_structures(m)) {
      tg.run([&m] { build_intersect_structures(m); });
      any = true;
    }
  if (any)
    tg.wait();
}

} // namespace tf::ts
```

- [ ] **Step 2: Resolve `same_data(...)` for the pair self-aliasing guard.** The pair overload must not build the same underlying `mesh_data` from two threads (a user can pass the same mesh, or two handles sharing one `_data`, to a pairwise op). Read `wasm_mesh.hpp` for how `_data` is stored (shared_ptr) and whether an identity accessor exists.
  - If `wasm_mesh` already exposes the underlying pointer (e.g. a `data()`/`raw()`/`get()` returning the `shared_ptr`/raw `mesh_data*`), implement `same_data` in terms of it.
  - Otherwise add a minimal `auto same_data(const wasm_mesh &o) const -> bool { return _data == o._data; }` to `wasm_mesh.hpp` (shared_ptr equality compares the controlled object). Add it in Task 1's commit or here — your call, but it MUST exist before this header compiles.
  - Confirm `_data`'s type supports `==` (shared_ptr does). Report what you found and which path you took.

- [ ] **Step 3: Compiles via Task 8.** No separate build.

- [ ] **Step 4: Commit.**

```bash
git add typescript/cpp/include/trueform/ts/core/build_intersect_structures.hpp typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp
git commit -m "feat(ts): parallel intersect-structure build helper (per-mesh/pair/N)"
```

---

## Task 3: Wire boolean (PAIR)

**Files:**
- Modify: `typescript/cpp/src/cut/boolean_impl.hpp` (`sync_boolean` ~line 49, `sync_boolean_with_curves` ~line 83)

- [ ] **Step 1: Add the include** near the existing includes (alphabetical within the `trueform/ts/core/...` group):

```cpp
#include "trueform/ts/core/build_intersect_structures.hpp"
```

- [ ] **Step 2: In `sync_boolean`, insert the parallel build** immediately after the `has0`/`has1` reads and BEFORE the first structure getter (`auto fm0 = m0.face_membership_range();`, line ~54):

```cpp
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  build_intersect_structures(m0, m1);
  auto fm0 = m0.face_membership_range();
```

- [ ] **Step 3: Same insertion in `sync_boolean_with_curves`** (after `has0`/`has1`, before `auto fm0 = m0.face_membership_range();` at ~line 89).

The subsequent `face_membership_range()`/`manifold_edge_link_range()`/`tree()` calls are now cache hits — no other change. The `_with_curves` and per-op (`union`/`intersection`/`difference`) and async wrappers all funnel through these two sync drivers, so they're covered transitively.

- [ ] **Step 4: Build + test via Task 8.** Commit:

```bash
git add typescript/cpp/src/cut/boolean_impl.hpp
git commit -m "perf(ts): parallel structure build in boolean dispatch"
```

---

## Task 4: Wire polygon arrangements (SELF)

**Files:**
- Modify: `typescript/cpp/src/cut/polygon_arrangement_impl.hpp` (`sync_polygon_arrangements` ~line 45, `sync_polygon_arrangements_with_curves` ~line 57)

- [ ] **Step 1: Add the include** (alphabetical, with the other `trueform/ts/core/...` includes).

- [ ] **Step 2: In each sync driver, insert before the first structure getter** (`auto ... = m.face_membership_range();`, lines ~49 / ~61):

```cpp
  build_intersect_structures(m);
  auto fm = m.face_membership_range();
```

(Single mesh → `parallel_invoke(tree, {fm;mel})` overlaps the tree with the topology within the one mesh.)

- [ ] **Step 3: Build + test via Task 8.** Commit:

```bash
git add typescript/cpp/src/cut/polygon_arrangement_impl.hpp
git commit -m "perf(ts): parallel structure build in polygon arrangement dispatch"
```

---

## Task 5: Wire mesh arrangements (N-MESH)

**Files:**
- Modify: `typescript/cpp/src/cut/mesh_arrangement_impl.hpp` (`run_arrangement` ~line 80, `run_arrangement_with_curves` ~line 126)

- [ ] **Step 1: Add the include** (alphabetical with the `trueform/ts/core/...` group).

- [ ] **Step 2: Read `run_arrangement`** to find where the `meshes` vector is in scope and where the per-mesh getters fire (inside the `tf::make_mapped_range(...)` closure, ~lines 87–94). Insert the parallel build BEFORE the `any_transformed` scan / mapped_range construction — i.e. as the first statement that uses `meshes`:

```cpp
  build_intersect_structures_all(meshes);
```

- [ ] **Step 3: Same in `run_arrangement_with_curves`** (before its mapped_range, ~line 142).

This pre-builds every mesh's tree/fm/mel across all meshes in parallel; the mapped_range closure's getters are then cache hits. The sync (`sync_mesh_arrangement*`) and async (`async_mesh_arrangement*`) entry points both delegate to these two core drivers, so both are covered. The `meshes` vector is already materialized on the main thread by `extract_meshes` and (for async) moved into the worker lambda, so it's safe to fan out over.

- [ ] **Step 4: Build + test via Task 8.** Commit:

```bash
git add typescript/cpp/src/cut/mesh_arrangement_impl.hpp
git commit -m "perf(ts): parallel structure build in mesh arrangement dispatch"
```

---

## Task 6: Wire the intersect module (PAIR + N-MESH + SELF)

**Files:**
- Modify: `typescript/cpp/src/intersect/intersect_impl.hpp`

- [ ] **Step 1: Add the include** (alphabetical with the `trueform/ts/core/...` group).

- [ ] **Step 2: `sync_intersection_curves` (PAIR, ~line 35).** After `has0`/`has1`, before the first `m0.face_membership_range()` (~line 39):

```cpp
  build_intersect_structures(m0, m1);
```

- [ ] **Step 3: `intersection_curves_list_impl` (N-MESH, ~line 100).** Before the `any_transformed` scan / mapped_range (~line 105), as the first statement using `meshes`:

```cpp
  build_intersect_structures_all(meshes);
```

- [ ] **Step 4: `sync_self_intersection_curves` (SELF, ~line 148).** Before the first `m.face_membership_range()` (~line 151):

```cpp
  build_intersect_structures(m);
```

- [ ] **Step 5: Do NOT touch `sync_isocontours` / `sync_isocontours_multi`** — they pass only `polygons_range()` + `scalars`, never tree/fm/mel (verified). Leave them.

The async wrappers (`async_intersection_curves`, `async_intersection_curves_list`, `async_self_intersection_curves`) forward to these sync drivers, so they're covered.

- [ ] **Step 6: Build + test via Task 8.** Commit:

```bash
git add typescript/cpp/src/intersect/intersect_impl.hpp
git commit -m "perf(ts): parallel structure build in intersect dispatch (pair/list/self)"
```

---

## Task 7 (OPTIONAL — confirm scope): Wire embedded curves (PAIR + SELF)

> The user's explicit list was boolean + polygon arrangements + mesh arrangements + intersect module. `embedded_impl.hpp` (cut module) has the **identical** structure-getter pattern and is a sibling gold path. Include this task only if you want embedded covered too; otherwise skip and note it.

**Files:**
- Modify: `typescript/cpp/src/cut/embedded_impl.hpp`

- [ ] **Step 1: Add the include.**
- [ ] **Step 2: `sync_embedded_intersection_curves` + `_with_curves` (PAIR, ~lines 49 / 85).** After `has0`/`has1`, before the first `m0.face_membership_range()` (~L54 / ~L90): `build_intersect_structures(m0, m1);`
- [ ] **Step 3: `sync_embedded_self_intersection_curves` + `_with_curves` (SELF, ~lines 124 / 137).** Before the first getter (~L128 / ~L141): `build_intersect_structures(m);`
- [ ] **Step 4: Commit.**

```bash
git add typescript/cpp/src/cut/embedded_impl.hpp
git commit -m "perf(ts): parallel structure build in embedded curve dispatch"
```

---

## Task 8: Build the wasm bundle and run the TS test suite

**Files:** none (verification).

- [ ] **Step 1: Build the wasm bundle.** From `typescript/`:

```bash
node build.mjs
```
This runs `emcmake cmake -S <root> -B build-wasm -DTF_BUILD_TYPESCRIPT=ON -DCMAKE_BUILD_TYPE=Release` (first time) then `cmake --build build-wasm --target trueform_wasm --parallel`, then bundles via esbuild.
Expected: configures + compiles + links `trueform_wasm` with no errors. (Confirms Tasks 1–7 compile: the new builders, the helper, TBB usage, and the wired call sites.)

- [ ] **Step 2: Run the full TS test suite.**

```bash
node tests/run.mjs
```
Expected: all tests pass — the run prints only failures + a summary. Pay special attention to:
- `test_cut.mjs` — boolean, polygon/mesh arrangement, embedded results unchanged.
- `test_intersect.mjs` — intersection curves (pair/list/self), isocontours unchanged.
- `test_async_precompute.mjs` — the async/precompute path (this exercises the async wrappers that now spawn TBB inside a pthread worker — nested parallelism; this is the highest-risk interaction, so it MUST pass).

If a test fails or the build errors, STOP and report the exact message + file. The most likely failure modes: (a) `same_data` accessor not found (Task 2 Step 2 unresolved), (b) nested-TBB-on-pthread-worker issue surfacing in `test_async_precompute.mjs`, (c) a missing include of the new helper at a call site.

- [ ] **Step 3: (Optional) Confirm the speedup.** If there's a cut/intersect timing test or `test_bench_spatial.mjs`-style harness, run it before/after to confirm first-op latency on fresh meshes improves. Not required for correctness.

- [ ] **Step 4: Final commit (if any verification-only fixes were needed).** Otherwise the per-task commits stand.

---

## Self-Review

**1. Scope coverage:** boolean (Task 3) ✓, polygon arrangements (Task 4) ✓, mesh arrangements (Task 5) ✓, entire intersect module — pair/list/self (Task 6) ✓; isocontours correctly excluded ✓; embedded optional (Task 7) ✓. Helper + builders (Tasks 1–2) ✓. Build+test (Task 8) ✓.

**2. Placeholder scan:** none. The `~line N` markers are "read to confirm exact line" instructions, not placeholders — each names the precise function and the anchor statement (the first `face_membership_range()` / the mapped_range / the `meshes` vector) to insert before.

**3. Type consistency:** helper names `build_intersect_structures` (per-mesh), `build_intersect_structures(a,b)` (pair, with `same_data` guard), `build_intersect_structures_all(std::vector<wasm_mesh<Real>>&)` (N), `needs_intersect_structures` (the fresh-gate predicate) — used consistently across Tasks 3–7. `build_tree()`/`build_face_membership()`/`build_manifold_edge_link()` are the wasm_mesh void builders from Task 1. mel-after-fm ordering encoded in the per-mesh helper. Thread-safety + caching rely on the verified `ensure_*` behavior.

**4. Cache gating (verified):** `is_tree_fresh()`/`is_face_membership_fresh()`/`is_manifold_edge_link_fresh()` exist on `wasm_mesh` (wasm_mesh.hpp:181-196 → `_data`) and equal the exact `ensure_*` skip condition (`_X.is_valid() && _X_gen == _faces_gen`, mesh_data.hpp:288/300/305). The helper gates every dispatch on `!is_*_fresh()`, so nothing is built or dispatched when fresh. No re-verification needed by the executor — confirmed.

**Known verification points the executor MUST resolve (flagged, not guessed):** (a) the `same_data`/`_data`-identity accessor on `wasm_mesh` (Task 2 Step 2) — read and implement; (b) exact insertion lines via the named anchors before editing; (c) nested-TBB-on-pthread behavior under `test_async_precompute.mjs` (Task 8 Step 2).
