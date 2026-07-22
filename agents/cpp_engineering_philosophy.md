# C++ Engineering Philosophy and Code Patterns

This document captures what makes trueform fast, robust, and maintainable — the engineering principles behind the code. An agent that internalizes these patterns should produce code indistinguishable from the existing codebase.

---

## 1. Parallelism as the Default

Everything is parallel unless there's a reason not to be.

### 1.1 The Algorithm Vocabulary

| Algorithm | When to Use | TBB Underneath |
|-----------|-------------|----------------|
| `tf::parallel_for_each(r, f)` | Per-element work | `tbb::parallel_for(blocked_range)` |
| `tf::parallel_for(r, f)` | Subrange work `f(begin, end)` | `tbb::parallel_for(blocked_range)` |
| `tf::parallel_copy(in, out)` | Bulk copy | `tbb::parallel_for(blocked_range<size_t>)` |
| `tf::parallel_fill(r, val)` | Initialize buffer | via `parallel_for` |
| `tf::parallel_iota(r, start)` | Sequential IDs | via `parallel_for` |
| `tf::parallel_transform(in, out, f)` | Element-wise transform | `tbb::parallel_for(blocked_range<size_t>)` |
| `tbb::parallel_sort(begin, end, cmp)` | Sorting | TBB merge sort |
| `tbb::task_group` | Heterogeneous parallel tasks | Fork-join |
| `tbb::parallel_invoke(f0, f1, ...)` | Fixed number of tasks | Fork-join |

**Never** use `std::copy`, `std::fill`, `std::iota`, or raw loops for bulk operations. Always use the `tf::parallel_*` equivalents.

### 1.2 The `tf::checked` Fallback

```cpp
tf::parallel_for_each(range, func, tf::checked);
```

Sequential fallback for ranges < 1000 elements. Use `tf::checked` when the range size is unpredictable. Omit it when you know the range is large (e.g., iterating all faces of a mesh).

### 1.3 Sentinel-Based Lazy Discovery

The most distinctive parallel pattern in trueform. Used throughout `cut/construct/`, `reindex/`, and `clean/`.

```cpp
// Allocate map, fill with sentinel
tf::buffer<Index> point_map;
point_map.allocate(n_points);
const Index sentinel = n_points;
tf::parallel_fill(point_map, sentinel);

// Discover vertices: first encounter writes, subsequent reads
Index curr = 0;
for (auto v : vertices) {
    if (point_map[v] == sentinel) {
        point_map[v] = curr++;
        discovered_ids.push_back(v);
    }
}
```

**Why not a hash map?** `buffer[n]` with sentinel is O(1) lookup, cache-friendly, no hash overhead, and no collision handling. The sentinel value is always the buffer size itself (guaranteed out of range).

**Clearing is O(discovered)**: Walk only the discovered entries, not the entire buffer.

```cpp
for (auto vi : discovered_ids)
    point_map[vi] = sentinel;  // Ready for next component
```

Real example from `make_arrangement_map_data.hpp`:
```cpp
for (const auto &v : loop) {
    if (v.source == tf::intersect::graph::vertex_source::original) {
        auto flat = d.point_offsets[desc.tag] + v.id;
        if (d.original_map[flat] == sentinel) {
            d.original_map[flat] = curr++;
            ids.push_back(v.id);
        }
    }
}
```

### 1.4 Thread-Local Aggregation (Lock-Free)

When parallel tasks produce variable-length output, use `tf::local_buffer<T>`:

```cpp
tf::local_buffer<intersection_t> l_intersections;
l_intersections.reserve_all(1000);

// Inside parallel tasks:
l_intersections.push_back({...});  // Thread-local, no lock

// After barrier:
auto merged = l_intersections.to_buffer();
```

Each thread's buffer lives in a `cache_aligned_slot` (128-byte alignment) to prevent false sharing. Merge after the barrier is a sequential copy.

For scalar aggregation, use `tf::local_value<T>`:

```cpp
tf::local_value<int> local_count;
// In parallel: (*local_count)++;
int total = local_count.aggregate(std::plus<>{});
```

### 1.5 Benign Race: Idempotent Bool Writes

When marking which vertices/faces are "used", multiple threads can write `true` to the same location without synchronization:

```cpp
tf::buffer<bool> point_mask;
point_mask.allocate(n_points);
tf::parallel_fill(point_mask, false);

// benign race: multiple threads may write `true` to the same byte.
// safe because writes are idempotent and there's a barrier at loop end.
tf::parallel_for_each(
    tf::make_indirect_range(face_im.kept_ids(), polygons.faces()),
    [&](auto &&face) {
        for (auto &e : face)
            point_mask[e] = true;
    },
    tf::checked);
```

### 1.6 `tbb::task_group` for Heterogeneous Work

When different tasks operate on different data (e.g., per-mesh construction):

```cpp
tbb::task_group tg;
for (Index t = 0; t < n_meshes; ++t) {
    tg.run([&, t] {
        // Per-mesh work: copy points, fill labels, etc.
        tf::parallel_copy(
            tf::make_indirect_range(map_data.original_ids[t], forms[t].points()),
            pts_range[t]);
    });
}
tg.run([&] {
    // Separate task for intersection points
    tf::parallel_copy(ig_points_range, tf::drop(pts_buf, offset));
});
tg.wait();
```

### 1.7 Parallel Data Generation

These are the core patterns for generating variable-length output in parallel. They drive performance in topology construction, intersection computation, and mesh assembly.

**`tf::generic_generate(input, output, lambda)`** — Each element produces variable output. Thread-local buffers merged automatically. Use instead of manual loops + push_back.

**`tf::generate_offset_blocks(input, output, lambda)`** — Each element produces one variable-length block. Builds offset_block_buffer directly.

**`tf::blocked_reduce(input, global, local_init, accumulate, merge)`** — Parallel accumulation with thread-local state. `accumulate` runs in parallel on chunks, `merge` runs sequentially to combine. Use when generating complex multi-buffer output.

**`tf::blocked_reduce_sequenced_aggregate(...)`** — Like `blocked_reduce` but merge order matches input order. Use when output ordering must be preserved.

### 1.8 Sort-Then-Group

A recurring workflow for partitioning by labels:

```cpp
// 1. Create IDs
tf::buffer<Index> ids;
ids.allocate(n);
tf::parallel_iota(ids, Index(0));

// 2. Sort by label
tbb::parallel_sort(ids.begin(), ids.end(),
    [&](auto a, auto b) { return labels[a] < labels[b]; });

// 3. Find group boundaries
tf::buffer<Index> offsets;
tf::compute_offsets(ids, std::back_inserter(offsets), Index(0),
    [&](auto a, auto b) { return labels[a] == labels[b]; });

// 4. Per-group views
auto groups = tf::make_offset_block_range(offsets, ids);

// 5. Process groups in parallel
tbb::task_group tg;
for (auto group : groups)
    tg.run([&, group = tf::make_range(group)] { /* ... */ });
tg.wait();
```

---

## 2. Range Composition — Zero-Copy Data Pipelines

### 2.1 The Philosophy

Data flows through lazy transformations. No intermediate buffers unless explicitly materialized. Every `make_*_range` returns a lightweight view object that transforms on iteration.

### 2.2 Composition Patterns

**Nested gather + remap** (reindexing faces with remapped vertex IDs):
```cpp
tf::parallel_copy_blocked(
    tf::make_indirect_range(
        face_im.kept_ids(),
        tf::make_block_indirect_range(polygons.faces(), point_im.f())),
    out.faces());
```
Read inside-out:
1. `block_indirect_range(faces, point_im.f())` — each face's vertex indices remapped through point_im
2. `indirect_range(kept_ids, ...)` — select only kept faces
3. `parallel_copy_blocked` — materialize to output

**Per-mesh lazy pipeline** (N-mesh arrangement construction):
```cpp
auto uncut_faces = tf::make_mapped_range(
    tf::make_sequence_range(n_meshes),
    [&](Index t) {
        return tf::make_indirect_range(
            map_data.original_face_ids[t],
            tf::make_block_indirect_range(
                forms[t].faces(),
                tf::make_mapped_range(original_maps[t],
                    [off = offsets[t]](Index x) { return x + off; })));
    });
```
A "range of ranges" — `uncut_faces[t]` lazily constructs mesh t's face pipeline.

**Buffer partitioning via take/drop**:
```cpp
tf::parallel_copy(original_points, tf::take(output, n_original));
tf::parallel_copy(created_points, tf::drop(output, n_original));
```

**Zip + offset_block_range for grouped iteration**:
```cpp
for (auto zipped : tf::make_offset_block_range(
        fc.tag_offsets(), tf::zip(fc.descriptors(), fc.loops()))) {
    for (auto [desc, loop] : zipped) { /* per-face work */ }
}
```

**Slide range for consecutive pair walking**:
```cpp
for (auto [a, b] : tf::make_slide_range<2>(path)) {
    half_edges[a].next = b;
    half_edges[b].prev = a;
}
```

### 2.3 Materialization

Lazy ranges become concrete via `parallel_copy`:
```cpp
tf::buffer<Index> output;
output.allocate(range.size());
tf::parallel_copy(range, output);
```

### 2.4 Static Size Propagation

`tf::static_size_v<Range>` flows through composition. If you `make_blocked_range<3>(buffer)`, the result has static size 3. `make_indirect_range(ids, data)` inherits the static size of `ids`. This enables compile-time loop unrolling and specialization.

---

## 3. The View/Buffer Split

### 3.1 Principle

**Buffers own memory. Views reference it.** Functions accept views, callers manage buffer lifetime.

```
polygons_buffer<int, float, 3, 3>   ← owns faces + points
    .polygons()                      → tf::polygons<...>  (view)
    .points()                        → tf::points<...>    (view)
    .faces()                         → tf::faces<...>     (view)
```

### 3.2 The `.polygons()` Accessor

Every `*_buffer` type has an accessor that returns the corresponding view:
- `polygons_buffer::polygons()` → `tf::polygons<Policy>`
- `points_buffer::points()` → `tf::points<Policy>`
- `segments_buffer::segments()` → `tf::segments<Policy>`

Views are what you pass to algorithms. Buffers are what you store.

### 3.3 Element Views

Individual elements within buffers are accessed as views:
- `points[i]` → `point_like<Dims, pt_view<T, Dims>>` (non-owning, mutable)
- `points[i]` on a const view → `point_like<Dims, pt_view<const T, Dims>>` (read-only)

---

## 4. Policy Composition

### 4.1 The `tag()` / `operator|` Pattern

```cpp
auto form = polygons | tf::tag(tree) | tf::tag(membership);
```

Each `tag()` wraps the current policy in a new layer. The data stays where it is — no copies. The tagged form carries additional structures for algorithms that need them.

### 4.2 Compile-Time Feature Detection

```cpp
if constexpr (tf::has_tree_policy<decltype(form)>) {
    // Tree is available — use it
} else {
    // No tree — build one or skip
}
```

Zero runtime cost when features are absent. The dead branch is eliminated at compile time.

### 4.3 `tf::none_t` for Zero-Cost Optionals

```cpp
template <typename Index = tf::none_t, typename Policy>
auto triangulated(const tf::polygons<Policy> &polygons) {
    if constexpr (std::is_same_v<Index, tf::none_t>) {
        using LocalIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
        return triangulated<LocalIndex>(polygons);
    } else {
        // Actual implementation with concrete Index type
    }
}
```

When the user doesn't specify `Index`, the function deduces it from the input. No runtime dispatch, no virtual calls.

---

## 5. The `build()` Pattern

Heavyweight data structures are default-constructed empty, then populated:

```cpp
tf::aabb_tree<int, float, 3> tree;
tree.build(polygons, tf::config_tree(4, 4));

tf::half_edges<int> he;
he.build(polygons);

tf::intersection_graph<int, tf::exact::int32> ig;
ig.build(ibp, apply_to_face, get_mesh_point, mode);
```

**Why**: Separates allocation from computation. Enables reuse (call `build()` again with different data). Enables profiling of build time. Matches TBB patterns where structures are built once and queried many times.

**Accessors after build**: `.points()`, `.faces()`, `.loops()`, `.descriptors()`, etc.

---

## 6. Exact Arithmetic

### 6.1 Int32 Coordinate Space

Float coordinates are converted to int32 using 99% of INT_MAX range:
```cpp
auto converter = tf::make_pt_converter<tf::exact::int32>(polygons);
auto int_point = converter(float_point);
auto float_point = converter.deconvert(int_point);
```

### 6.2 Cascading Precision

`meta<int32>::T1 = int64`, `meta<int32>::T2 = int128`. Intermediate products use wider types to prevent overflow:
```cpp
// orient3d_value uses T2 (int128) for the 3x3 determinant
auto vol = orient3d_value<int32>(p0, p1, p2, p3);  // returns int128
```

### 6.3 SoS (Simulation of Simplicity)

When the determinant is exactly zero, SoS uses globally unique vertex IDs to break ties deterministically:
```cpp
auto result = orient3d_sos(vertices);  // NEVER returns zero
```

This eliminates all geometric degeneracy in boolean operations and intersections.

---

## 7. Naming and Formatting

### 7.1 Names

- **Functions**: `snake_case` — `make_boundary_edges`, `fit_icp_alignment`
- **Types**: `snake_case` — `polygons_buffer`, `half_edges`, `aabb_tree`
- **Template parameters**: `Index`, `RealT`, `Int`, `Dims`, `Ngons`, `Policy`, `Range`
- **Files**: `snake_case.hpp`, one primary function/class per file
- **No type aliases in library code** — use full names for grepability
- **Trailing return types**: `auto foo(...) -> ReturnType` — used consistently

### 7.2 Header Layout

```cpp
/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 * ...
 */
#pragma once
#include "dependency.hpp"          // IWYU: only what's needed

namespace tf {

/// @ingroup module_subgroup
/// @brief One-line description.
/// @tparam Index ...
/// @param polygons ...
/// @return ...
template <typename Index = tf::none_t, typename Policy>
auto function_name(const tf::polygons<Policy> &polygons) {
    // implementation
}

} // namespace tf
```

### 7.3 Umbrella Headers

```cpp
// geometry.hpp
#include "./geometry/make_box_mesh.hpp"      // IWYU pragma: export
#include "./geometry/compute_normals.hpp"    // IWYU pragma: export
// ...
```

### 7.4 `.clang-format`

`BasedOnStyle: LLVM`

---

## 8. Memory Discipline

- **No raw `new`/`delete`** — ever
- **`tf::buffer<T>`** for POD arrays (trivially destructible only)
- **`tf::small_vector<T, N>`** for small bounded collections
- **`std::vector<T>`** only for non-POD types
- **Reserve, then push** — `buffer.reserve(n)` then `push_back()`, not `allocate(n)` + assign
- **Allocate exact** — when size is known: `buffer.allocate(n)` then write directly
- **Sentinel maps over hash maps** — `buffer[n]` with sentinel for integer-keyed lookups

---

## 9. The Sort Shape — identity without associative structures

The house replacement for every hash map, keyed by access pattern
(law: `cpp_performance_philosophy.md` §1):

- **Sorted flat table + `equal_range`/`lower_bound`** — global tables
  (splits by edge key, merges by from-key). Sort once, binary-search
  always; apply transitive closure at consolidation so lookups never
  chain.
- **Scatter array + generation stamp** — dense bounded id space reused
  across items without clearing.
- **Small linear scan** — population bounded by a tiny constant
  (originals on a face = its corners).
- **The flat lift** for mixed id spaces: created ids keep their id,
  originals lift by a per-tag dense base — one integer space, no
  per-vertex remapping.

The full pipeline shape: parallel generate records → `tbb::parallel_sort`
by the canonical key → ONE sequential adjacency sweep (dedup happens
here, before any geometry exists) → parallel materialize/apply.
Duplicates from different producers become literally equal records and
fuse at the sweep. Determinism is structural: sorted order is canonical
order.

**Data identity lives in parameters, not carried geometry.** A split is
a dyadic t on its edge (numerator over 2^k); every carrier
materializes the same lattice point from the same parameter via one
rounding. Clamp parameters into range at registration — rounding can
push a materialized point past an endpoint, and unclamped means
extrapolation. A changed global fact (a weld, a merge) is a
SUBSTITUTION over already-computed output plus dropping what collapsed
— never a recomputation (law §2).

## 10. Parallel-State Discipline

- **`local_t` stays LIGHT.** Block-local state travels BY VALUE through
  the flow graph (work → sequencer → aggregate); heavy members get
  deep-copied per block. Heavy reusable scratch lives thread-local
  (`tf::local_value` / `tf::local_buffer`); `local_t` carries a pointer.
- **Propose, then materialize.** Threads emit records into local
  buffers; anything that assigns ids or grows shared tables runs once,
  serially, over the sequenced aggregate.
- **Queues over mutable structures need generation stamps.** A queued
  reference (face, slot) can alias a different entity after a rebuild;
  validate the stamp at pop and requeue through the discovery path so
  the fact is genuinely rediscovered.
- **Guard parity between sibling paths.** Build and refine, stock and
  recovery — paths over the same structure must skip the same members
  (dead loops, frozen edges) and validate the same staleness. An
  asymmetric guard is a latent OOB or a silent semantic drift.
- Aggregations that append to a shared buffer other tasks read must
  stage locally and append after the reduce — the aggregator must
  never reallocate what a task may be reading.

## 11. Hot-Loop Mechanics

- `reallocate` + pointer write, never `push_back`, in tight loops.
- Per-FACE quantities memoized per face, not recomputed per region.
- Move bodies OUT of callbacks: the callback fetches what only it can;
  everything else runs flat.
- Gate clears: sweeping an empty structure per item is real cost —
  clear only when used.
- Fast paths stay branch-light; a per-edge lookup in the common no-hit
  case is measurable.

## 12. Benchmark Discipline

- mimalloc'd bench binaries, single-threaded AND parallel (parallel is
  the gate; single-thread parity can hide serial aggregation
  bottlenecks), best-of-N or medians, never one run.
- Outputs proven IDENTICAL before timing — winding-aware when
  orientation matters (rev swap + min-rotation canon; a full sort
  hides winding bugs).
- Fresh binaries: `touch` probe sources before rebuilding; count build
  errors AND warnings. Stage-by-stage structural benches beat
  end-to-end lumps; a path env (`TF_PATH=old|new`) keeps profiles
  clean.
- The DYLD-injected override mimalloc is for bench BINARIES only —
  trueform's own backend is explicit `mi_*` (`MI_OVERRIDE=OFF`), safe
  under any host runtime; the injected override under CPython is a
  second allocator instance and segfaults at import.
- Record refuted hypotheses next to the code or in memory — an
  unrecorded refutation gets re-attempted.

## 13. What NOT to Do

1. **Don't use type aliases in library code.** Write `tf::polygons_buffer<Index, RealT, Dims, 3>`, not `using Mesh = ...`. Full names are greppable.

2. **Don't add unnecessary abstractions.** Three similar lines of code is better than a premature helper function.

3. **Don't use raw loops for bulk operations.** Use `tf::parallel_for_each`, `tf::parallel_copy`, etc.

4. **Don't use `std::copy`/`std::fill`/`std::iota`.** Use `tf::parallel_copy`/`tf::parallel_fill`/`tf::parallel_iota`.

5. **Don't use hash maps for integer-keyed lookups.** Use sentinel-based buffer maps.

6. **Don't use mutexes or atomics in hot paths.** Use thread-local storage (`local_buffer`, `local_value`) and merge after barriers.

7. **Don't allocate per-iteration.** Reuse buffers across loop iterations. Clear by walking used entries, not by reallocating.

8. **Don't add docstrings/comments to code you didn't change.** Only comment where logic isn't self-evident.

9. **Don't add error handling for scenarios that can't happen.** Trust internal code. Only validate at system boundaries.

10. **Don't use `if constexpr` without `tf::none_t`.** The pattern is: default template parameter `= tf::none_t`, then `if constexpr (std::is_same_v<T, tf::none_t>)` to trigger deduction.

---

## 14. Portability (MSVC)

The library is developed on clang/AppleClang but must build on MSVC (Windows CI). These are compiler-specific traps that clang accepts silently and only fail on Windows — so they can't be caught by building locally. Treat them as hard rules.

1. **No local `constexpr` variable referenced inside a lambda body -- ANY reference, `if constexpr` conditions included.** MSVC rejects it ("read of a variable outside its lifetime"); proven 2026-07 on `tangential_relaxation.hpp` where the only use was `if constexpr (HasMask)` inside the lambda. Declare the `constexpr` *inside* the lambda body that uses it (preferred -- it stays a constant and there is nothing to capture), or hoist it to a runtime local before the lambda when several scopes share it. Template non-type parameters (e.g. `bool WantLabels`) are fine — they're constants from the enclosing template, not captured.
   ```cpp
   constexpr std::size_t N = ...;
   auto f = [&] { use(N); };        // BAD on MSVC (odr-uses constexpr local N)
   const std::size_t n = N;
   auto g = [&] { use(n); };        // OK — runtime local
   ```

2. **Capture structured-binding names via an explicit init-capture.** Pre-C++20 MSVC can't capture a structured binding through a default `[&]`/`[=]`. Rebind it in the capture list (or bind to a plain local first):
   ```cpp
   auto [a, b] = f();
   auto h = [&] { use(a); };        // BAD on MSVC (captures a structured binding)
   auto k = [&, &a = a] { use(a); }; // OK — explicit init-capture
   ```

3. **ASCII-only in Catch2 `TEST_CASE` / `SECTION` names.** ctest's test-name parsing chokes on non-ASCII (`→`, `×`, `°`, accented letters) on Windows. Keep registered test names ASCII; UTF-8 is fine in comments and code.

4. **No `M_PI` / `M_*` math macros.** They need `_USE_MATH_DEFINES` before `<cmath>` on MSVC. Use `tf::pi<T>` and the other `tf::` constants.
