# C++ Engineering Reference

This document contains implementation mechanics, code conventions, hot-loop
details, and portability rules. The design authorities are
`cpp_performance_philosophy.md` and `cpp_execution_patterns.md`; do not infer a
new phase shape from an isolated recipe here.

---

## 1. Parallelism at the Correct Grain

Parallelize the largest independent carrier. Small structural sweeps, prefixes,
leaf kernels, adjacency walks, and stateful inner loops are deliberately serial.

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

Use the `tf::parallel_*` equivalents for genuinely bulk independent work.
`std::copy`, raw loops, and small serial algorithms are correct inside
block-local kernels, ordered aggregation, prefix/rebase phases, and below
measured dispatch thresholds.

### 1.2 The `tf::checked` Fallback

```cpp
tf::parallel_for_each(range, func, tf::checked);
```

Pass `tf::checked` when range length is a sound proxy for total work and the
small-range case should stay serial. This keeps one bulk algorithm and lets the
primitive own its serial cutoff; do not open-code a second size branch around
the same operation. Do not apply it mechanically: a short range of large
polygons, deep blocks, or other expensive tasks should remain parallel when
each element costs more than the threading overhead.

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

### 1.4 Partition-Carried State

Variable output normally belongs to the partitioning primitive, not an
arena-indexed container:

```cpp
struct local_t {
    tf::buffer<record_t> records;
    scratch_t scratch;
};

tf::blocked_reduce(input, output, local_t{},
    [](auto block, local_t &local) {
        for (const auto &element : block)
            emit(element, local.records, local.scratch);
    },
    append_local);
```

Use stateful `parallel_for_each`, `generic_generate`, `sequenced_generate`,
`generate_offset_blocks`, or a blocked reduction according to output shape.
The local state is constructed once per task/block and reused by its serial
inner loop.

`local_buffer`, `local_vector`, and `local_value` are reserved for irregular
parallel tree traversal whose callback/task API cannot receive block state.

### 1.5 Idempotence Does Not Make a Race Benign

Two workers writing the same scalar location without synchronization is a C++
data race even when both store the same value and a barrier follows. Prefer a
phase shape that assigns each output slot to one worker: count/prefix/write,
group by target ID, block-local marks followed by a merge, or derive the mask
from an already unique carrier. Use atomics only when ownership cannot be
manufactured and the irregular concurrent claim is the intended mechanism.

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

**`tf::generic_generate(input, output, lambda)`** — Each element produces variable output. Per-block buffers are merged in unspecified completion order. Use when a later canonical sort makes input order irrelevant.

**`tf::sequenced_generate(input, output, lambda)`** — Variable output appended in input-block order. Use when flat output order itself carries positional meaning.

**`tf::generate_offset_blocks(input, output, lambda)`** — Each element produces one variable-length block. Builds offset_block_buffer directly.

**`tf::blocked_reduce(input, global, local_init, accumulate, merge)`** — Parallel accumulation with block-local state. `accumulate` runs over sequential chunks, and `merge` runs serially in unspecified completion order.

**`tf::blocked_reduce_sequenced_aggregate(...)`** — Parallel block work with aggregation in input-block order. Use when order constructs offsets, preserves `input[i] -> output block[i]`, keeps correlated jagged carriers aligned, or permits direct ID rebasing. Determinism is a consequence, not its only purpose.

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

// 5. Process independent groups in parallel
tf::parallel_for_each(groups, [&](auto group) { /* tight serial kernel */ });
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

When fixed arity is part of a function's contract, express it directly in the
parameter as `tf::range<Iterator, N>`. Do not add an `enable_if` layer around a
generic range merely to rediscover the static size; the direct signature is
clearer and more portable.

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
        return triangulated<
            std::decay_t<decltype(polygons.faces()[0][0])>>(polygons);
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

Classes own state, lifetime, invariants, and phase wiring. Express coherent
reusable algorithms as free functions with ranges, buffers, policies,
callbacks, and result carriers passed explicitly. Methods connect those
operations and commit their results; retain algorithmic work inside the class
only when it is inseparable from the owner's invariant.

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
- **Files**: `snake_case.hpp`, named for their primary operation or class
- **Implementation types remain explicit** — do not introduce convenience
  aliases that conceal concrete carrier or result types and reduce
  grepability. Public semantic aliases and required interface or trait aliases
  remain valid.
- **Trailing return types**: `auto foo(...) -> ReturnType` — used consistently

### 7.2 File and Namespace Ownership

**A first-level module directory holds only the module's user-facing
interface.** Headers directly under `include/trueform/<module>/` define `tf::`
symbols. Every symbol in a nested namespace — `tf::<module>`,
`tf::<module>::detail` — lives in a nested directory under that module, never
beside the public headers. A top-level module header that opens
`namespace tf::<module>` is misplaced: either the file belongs in a nested
directory, or its nested-namespace block must be factored into a
nested-directory header that the public header includes. Reading the top level
of a module directory should show the library's surface and nothing else.

**Classes stay thin; the work lives in free functions.** Factor every
self-contained step out of a class into a free function in the module's nested
namespace, one operation per header under a nested directory. A class body is
composition, ownership, and lifetime — not the algorithms it runs. Free
functions are how a second owner reuses a step without inheriting the first
owner's state, so the factoring is a reuse mechanism, not tidying. A class that
has grown to thousands of lines is a factoring failure rather than a large
problem: split it by operation until each header states one thing.

- In ordinary modules, headers directly under `include/trueform/<module>/`
  define that module's top-level `tf::*` surface. Nested internal headers do not
  introduce new top-level `tf::*` symbols; they live under the module-owned
  namespace `tf::<module>`. Further namespace nesting is an explicit domain
  choice, not something derived mechanically from every directory component.
- Internal owner classes live on the nested module implementation surface.
  Public nested namespaces and organizational exceptions are explicit,
  documented choices.
- Keep one coherent operation family or owner type per header. Only inseparable
  small implementation helpers share it; an independently reusable helper gets
  its own named header.
- `core/` is partitioned into nested directories because it is large. Its
  established public partitions, including `core/algorithm/` and `core/views/`,
  may still expose top-level `tf::*`; do not invent namespaces mechanically
  from those paths or generalize the exception to internal module directories.

### 7.3 Header Layout

Every header opens with the repository copyright/license block, copied verbatim
from a neighboring header in the same module — never retyped from memory and
never omitted. The skeleton below starts after that block; a new file whose
first line is `#pragma once` is incomplete.

```cpp
/* ... repository copyright/license block, copied from a neighbor ... */
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

### 7.4 Umbrella Headers

```cpp
// geometry.hpp
#include "./geometry/make_box_mesh.hpp"      // IWYU pragma: export
#include "./geometry/compute_normals.hpp"    // IWYU pragma: export
// ...
```

### 7.5 `.clang-format`

`BasedOnStyle: LLVM`

---

## 8. Memory Discipline

- **No raw `new`/`delete`** — ever
- **`tf::buffer<T>`** for types satisfying its uninitialized,
  byte-relocatable storage contract
- **`tf::small_vector<T, N>`** for small bounded collections
- **`std::vector<T>`** when element construction or destruction is required
- **Reserve, then push** — for block-local variable output whose size is not known
- **Allocate exact** — when counts or offsets make size known, then write directly
- **Sentinel maps over hash maps** — `buffer[n]` with sentinel for integer-keyed lookups

Allocation lifetime is part of data shape. An exact refactor preserves
allocation points and count, retained capacity, scratch reuse, ownership, and
release timing unless a separate reviewed and measured change proves another
shape better.

---

## 9. The Sort Shape — identity without associative structures

The normal replacement for associative lookup, keyed by access pattern
(see `cpp_execution_patterns.md` §7):

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

- **State follows the partition.** Block-local state is constructed once per
  task and reused by the serial inner loop. It may own buffers, scratch
  structures, and correlated outputs when that construction/copy cost is
  appropriate per block. Do not move it to arena-local storage merely because
  it is substantial.
- **Arena-local state is exceptional.** Use `local_value`, `local_buffer`, or
  `local_vector` only for irregular parallel tree traversal whose callback/task
  API cannot receive block state.
- **Propose, then materialize.** Threads emit records into local
  buffers; anything that assigns ids or grows shared tables runs once,
  after discovery. Use a sequenced aggregate when input order preserves
  carrier identity or enables direct offsets/rebasing.
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

- Run one timed benchmark process at a time. Parallel benchmarks occupy the
  machine; overlapping benchmark processes invalidate timing.
- mimalloc'd bench binaries, single-threaded AND parallel (parallel is
  the gate; single-thread parity can hide serial aggregation
  bottlenecks), best-of-N or medians, never one run.
- Outputs proven IDENTICAL before timing — winding-aware when
  orientation matters (rev swap + min-rotation canon; a full sort
  hides winding bugs).
- Rebuild the requested target and verify the changed translation unit or
  dependent target was rebuilt; count build errors AND warnings.
  Stage-by-stage structural benchmarks beat end-to-end lumps.
- The DYLD-injected override mimalloc is for bench BINARIES only —
  trueform's own backend is explicit `mi_*` (`MI_OVERRIDE=OFF`), safe
  under any host runtime. Do not inject that override under CPython; it
  creates a second allocator instance in one process.
- Record refuted hypotheses in task memory or benchmark artifacts so they are
  not repeated. Production code and comments remain timeless.

## 13. What NOT to Do

1. **Don't hide implementation carriers behind convenience aliases.** Write
   concrete carrier and result types so repository search finds their use.
   Public semantic aliases and required interface or trait aliases are not
   convenience aliases.

2. **Don't add unnecessary abstractions.** Three similar lines of code is
   better than a premature helper function. Do not use pointer presence or
   `nullptr` as an operation-mode selector; pointers remain valid for borrowing,
   storage, and interop, while semantic variation uses a direct operation or an
   explicit existing tag or policy. Do not build machinery for hypothetical
   modes.

3. **Don't leave large independent carriers serial.** Use the appropriate
   parallel primitive. Tight block-local kernels, prefixes, graph walks, and
   small measured cases remain serial.

4. **Don't use serial library algorithms for genuinely bulk independent work.**
   `std::copy`, `std::fill`, and raw loops remain valid inside small local or
   serial aggregation phases.

5. **Treat hash maps as a measured exception.** First use a dense sentinel or
   generation map, sort/group offsets, a sorted sparse table, or tiny linear
   storage. A bounded integer domain is directly addressable.

6. **Do not synchronize what phases can separate.** Prefer block-local
   proposals, disjoint output ranges, frozen authorities, and post-barrier
   equivalence collapse. Narrow irregular traversals may use atomics when the
   mechanism requires concurrent claiming.

7. **Don't allocate per-iteration.** Reuse buffers across loop iterations. Clear by walking used entries, not by reallocating.

8. **Code and comments are timeless.** Names, branches, structure, and comments
   describe only the present mechanism and contract. Comment only a non-obvious
   reason, invariant, ownership rule, or contract; git history and task records
   own development and debugging narrative.

9. **Don't add error handling for scenarios that can't happen.** Trust internal code. Only validate at system boundaries.

10. **Use `tf::none_t` when absence triggers type deduction.** A default
    template parameter of `tf::none_t` plus `if constexpr` is the standard
    zero-cost optional-type pattern. Ordinary compile-time dispatch on policies,
    static sizes, and traits may also use `if constexpr` directly.

---

## 14. Portability (MSVC)

The library is developed on clang/AppleClang but must build on MSVC (Windows CI). These are compiler-specific traps that clang accepts silently and only fail on Windows — so they can't be caught by building locally. Treat them as hard rules.

1. **No local `constexpr` variable referenced inside a lambda body -- ANY reference, `if constexpr` conditions included.** MSVC rejects the captured lifetime. Declare the `constexpr` *inside* the lambda body that uses it (preferred -- it stays a constant and there is nothing to capture), or hoist it to a runtime local before the lambda when several scopes share it. Template non-type parameters (e.g. `bool WantLabels`) are fine — they're constants from the enclosing template, not captured.
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
