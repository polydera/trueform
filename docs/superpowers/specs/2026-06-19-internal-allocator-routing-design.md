# Internal Allocator Routing — Design

**Status:** approved (brainstorm), pending plan
**Branch:** `feature/internal-allocator` (worktree `/Users/ziga/trueform-allocator`)
**Date:** 2026-06-19

## Goal

Route every heap allocation that **trueform owns** through a single, swappable
allocator (`tf::allocate` / `tf::deallocate`), so that internal scratch can later
use a fast allocator (e.g. tbbmalloc) **without users preloading anything**. This
PR is **behaviour-neutral**: the default backend is `operator new` / `operator
delete`, so timings and semantics are unchanged. The actual fast-allocator swap
is out of scope (a later, gated one-line change to the backend body).

### Why now

Measured earlier this session: on the n-ary arrangement the allocator-sensitive
stages (`ig`/`fc`) gain 1.3–1.8× under mimalloc/tbbmalloc, and the gap widens on
dense workloads. To capture that for users transparently, all internal
allocations must flow through one place we can re-point. Today they flow through
`new[]`/`delete[]` (buffer), `ska2`'s default allocator (hash), `std::queue`, and
scattered `std::vector` — none swappable.

## The governing principle

> **trueform owns it ⇒ our allocator. The user's `std::vector` *type* ⇒ default.**

- Anything we allocate, own, and free — directly or by handing it off with a
  destructor **we** write (numpy capsule, VTK delete-method) — uses
  `tf::allocate`. We control the free, so it frees with `tf::deallocate`. This is
  safe precisely because the handoff is ours.
- The only things that stay on the default allocator are (1) the **outer
  `std::vector<…>` we return** to the user, so they receive a vanilla
  `std::vector` (the buffers *inside* are still ours and free themselves via
  their dtor), and (2) throwaway per-query `new[]` we make in Python solely to
  fill a small numpy result.

## Components

### 1. `include/trueform/core/memory.hpp` (new)

The single allocation surface.

**Naming (collision-checked):** `tf::allocate`/`tf::deallocate` (top-level) are
free — they are the user-facing seam. Taken and NOT reused: `tf::core::allocate`
(the 2-arg container-resize helpers in `core/allocate.hpp`, a different namespace)
and `tf::vector` (the linear-algebra vector). The internal vector alias is
therefore `tf::core::std_vector`.

```cpp
namespace tf {
// raw, void*/bytes so deallocate needs no type and is a function pointer
auto allocate(std::size_t bytes) -> void*;     // default: ::operator new(bytes)
auto deallocate(void* p) -> void;              // default: ::operator delete(p)
                                               // (not noexcept: void(*)(void*) callback)
// std::allocator-compatible adapter over the two above
template <typename T> struct allocator { /* allocate(n)/deallocate(p,n) */ };

namespace core {
// the one internal vector (tf::vector is the linalg vector)
template <typename T> using std_vector = std::vector<T, tf::allocator<T>>;
}
}
```

- `allocate`/`deallocate` are **bytes/`void*`**, not typed. Reasons: (a)
  `deallocate(void*)` needs no element type, (b) `&tf::deallocate` is a
  `void(*)(void*)` — exactly what VTK `USER_DEFINED` / a numpy capsule need.
- Default backend = global `operator new/delete`. Swapping to tbbmalloc is a
  later edit to these two function bodies only.
- `tf::allocator<T>` is a normal stateless std-allocator (equality always true).

### 2. `tf::buffer` storage → `tf::allocate`

`include/trueform/core/buffer.hpp`. Today: `std::unique_ptr<T[]> _data` +
`new T[n]` (lines 94, 244, 265). Buffer already `static_assert`s
`is_trivially_destructible<T>` (line 41) and `memcpy`s on copy/grow — so it is
already POD-oriented. Replace raw `new T[]` with `tf::allocate(n*sizeof(T))`
reinterpreted as `T*`, and free with `tf::deallocate`. No constructors/destructors
run (sound for trivially-destructible T). `release()` still returns `T*`; the
returned pointer is now `tf::allocate`'d and **must be freed with
`tf::deallocate`** — this is the new public contract.

### 3. `finder.hpp` scratch → `tf::allocate`

`include/trueform/topology/components/finder.hpp:142,147`:
`unique_ptr<std::atomic<label_t>[]>` + `new std::atomic<label_t>[size]`.
`std::atomic` is trivially destructible. Route through `tf::allocate`/`deallocate`.

### 4. `tf::hash_map` / `tf::hash_set` → default to `tf::allocator`

`include/trueform/core/hash_map.hpp`, `hash_set.hpp` alias to
`ska2::flat_hash_map/set<K[,V], …Ts>` (open-addressing → one slot array). Default
the trailing allocator template argument to `tf::allocator<…>`. Transparent to all
call sites (VTK `polydata.cpp` ×3, examples).

### 5. `std::queue` → `tf::buffer` + advancing head

`include/trueform/cut/arrangements/propagate_inclusion_bits.hpp:132`,
`std::queue<Index>` — strict-FIFO multi-source BFS, drained once. Replace with
`tf::buffer<Index> q; std::size_t head = 0;` where push = `q.push_back(x)` and pop
= `q[head++]`, loop while `head < q.size()`. Frees once at scope end. No
behaviour change (same visitation order).

### 6. `std::vector` split: internal → `tf::vector`, boundary → default

- **Internal-only scratch `std::vector`** (does not escape to the user) →
  `tf::vector<T>`. Examples: `cut/cut_graph.hpp:125`
  (`vector<hash_map<Index,Index>>`), expression-compile temporaries
  (`coalesce_words`, `compile_cluster_node`).
- **Boundary `std::vector` we return** → **stays `std::vector` (default)** so the
  user gets a vanilla container: `reindex/split_into_components`/`split_into_domains`
  returns, VTK `split_into_components` return, `reindex/range.hpp` return,
  `csg::expression` `any_of`/`all_of`/`make_children` returns (also cold).
- **Interop overloads** that take a user's `std::vector<T>&` by reference
  (`core/allocate.hpp`, `core/reallocate.hpp`, `core/algorithm/generate_offset_blocks.hpp`)
  → stay `std::vector` by definition (they operate on the caller's container).

### 7. The 14 release boundaries → `tf::deallocate`

The released pointer is now `tf::allocate`'d; every consumer must free with
`tf::deallocate`:

- **VTK (12):** `vtk/src/core/make_vtk_array.cpp` (×5),
  `make_vtk_points.cpp`, `make_vtk_normals.cpp`, `make_vtk_cells.cpp` (×2),
  `vtk/include/trueform/vtk/core/make_vtk_cells.hpp` (×1). All currently
  `SetArray(ptr, n, 0, VTK_DATA_ARRAY_DELETE)` → switch to
  `VTK_DATA_ARRAY_USER_DEFINED` + register `tf::deallocate` as the free function
  (`vtkAbstractArray` user-defined delete is `void(*)(void*)`).
- **Python (2):** `python/include/trueform/python/util/make_numpy_array.hpp:64`,
  `python/include/trueform/python/spatial/gather_ids.hpp:46`. The nanobind capsule
  destructor (currently `delete[]`) → `tf::deallocate`.

This is the **safety-critical** part: without it, a backend swap silently
corrupts the heap (delete[]/free on a tbbmalloc pointer). The default backend keeps
working today either way, so this PR is safe to land, but the frees MUST move in
the same PR so the backend is swappable thereafter.

### 8. Docs → the `tf::deallocate` contract

Update prose/examples that document `release()`→`delete[]`:
`docs/content/cpp/2.modules/01.core.md` ("use `delete []`"),
`docs/content/cpp/5.examples/1.mesh-assembly.md`, `CLAUDE.md:37`,
`ARTICLE_DATA_PATTERNS.md:26`, `STL_ARTICLE_REFERENCE.md:247`. New contract:
"a released pointer was allocated by trueform; free it with `tf::deallocate`."

## Out of scope

- The actual tbbmalloc/fast backend (later, gated; this PR only adds the seam).
- Per-buffer allocator templating (`core_buffer<T, Allocator>`) — superseded; one
  global swappable allocator + the `tf::deallocate` contract is simpler and solves
  the same release-safety.
- Python per-query `new[]` for numpy results (point coords, transform matrices) —
  pure throwaway scratch, left on default `new[]`/capsule `delete[]`.
- `std::shared_ptr`/`unique_ptr` that own *trueform objects* (trees, meshes,
  wrappers) in bindings — they own objects built on `tf::buffer`, so they ride the
  allocator transitively; no direct change.

## Constraints / invariants

- **Behaviour-neutral:** default backend = `operator new/delete`; no timing or
  semantic change. Existing test suites must pass unmodified.
- **Trivially-destructible only:** buffer's existing static_assert guarantees raw
  alloc/free + `void*` deallocate is sound (no dtors to run).
- **One free path:** every trueform-owned allocation frees via `tf::deallocate`
  (directly or through a destructor we wrote).

## Search scope (audited)

`include/`, `python/`, `vtk/`, `typescript/cpp/`, `examples/`, `docs/content/cpp/`.
TypeScript has zero `tf::buffer::release()` (all `shared_ptr<buffer>`); the
`async_dispatcher.hpp` `acc.release()` is a TBB accessor lock, excluded.

## Testing

- Existing Catch2 suites (core, cut, topology, spatial) + python/ts test suites
  pass unchanged (behaviour-neutral).
- New: a focused test that `tf::buffer<T>::release()` → `tf::deallocate` round-trips
  with no leak (run under ASan if available).
- VTK + numpy paths: a smoke test that builds an array via the move overload and
  destroys it (VTK array dtor / numpy array GC) under ASan — proves the
  USER_DEFINED / capsule free matches `tf::allocate`.
- Coding standards: dev-cpp (snake_case, `make_` factories, trailing return types,
  `#pragma once` + copyright header, namespace close comments).
