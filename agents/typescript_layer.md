# TypeScript/WASM Boundary Contract

> **Task-specific authority.** Read this after `working_method.md`,
> `cpp_performance_philosophy.md`, and `cpp_execution_patterns.md` when changing
> TypeScript, embind, WASM storage, or async dispatch. This document describes
> boundary invariants. It is not a fixed file-layout recipe; inspect the nearest
> current registration and `typescript/CMakeLists.txt` before adding sources.

The TypeScript layer does not reimplement Trueform. It owns user-facing types,
validation, disposal, and promise composition around native handles. Geometry,
topology, cache construction, and bulk work remain in C++.

## Boundary model

```text
JS TypedArray
-> explicit copy into WASM (`wasm_ndarray::from_js`)
-> shared native storage and semantic C++ ranges
-> Trueform computation, possibly on TBB workers
-> native result handles owning WASM buffers
-> borrowed TypedArray views into the WASM heap
```

For async operations:

```text
main JS thread validates and creates native handles
-> embind `dispatch_*` returns a status address
-> worker runs only C++/WASM-safe code
-> worker publishes completion with a WASM atomic
-> JS waits with `Atomics.waitAsync`
-> main JS thread calls generic `retrieve(slot)`
-> C++ converts the stored result to `emscripten::val`
-> TypeScript wraps returned handles
```

The worker/main-thread split is part of correctness, not presentation.

## 1. WASM ndarray ownership

`wasm_ndarray<T>` in
`typescript/cpp/include/trueform/ts/core/wasm_ndarray.hpp` stores:

- `std::shared_ptr<tf::buffer<T>>` for allocation ownership;
- a logical element offset and length;
- shape metadata local to that handle.

Consequences:

- `from_js(...)` allocates in WASM and copies a JS TypedArray into it.
- `from_buffer(...)` moves an existing Trueform buffer into shared WASM-owned
  storage without copying its elements.
- `row(...)` and `slice(...)` are zero-copy views. They share storage and adjust
  logical offset, length, and shape.
- `shallow_copy()` shares storage but owns independent shape metadata.
- `destroy()` resets that handle's shared ownership and is idempotent. Do not
  use the handle after destroying it.

### `data()` is borrowed

`wasm_ndarray::data()` returns an `emscripten::typed_memory_view`. The resulting
TypedArray points into the WASM heap; it does not own the native allocation.

Therefore:

- keep an owning `NDArray` or another shared native handle alive while using
  the view;
- do not treat a cached `.data` value as an ownership transfer;
- reacquire `.data` after operations that may grow WASM memory rather than
  assuming an old JavaScript view remains current;
- use `toArray()` or an explicit TypedArray copy when independent JavaScript
  ownership is required.

This is the most important distinction at the TypeScript memory boundary:
native handles own; TypedArray views borrow.

### Range construction inside bindings

`make_range()` covers the complete logical ndarray view and already includes
its offset. Prefer it for a full linear pass.

`raw_data()` also points at the logical start, not the allocation start. Use it
for strided, broadcast, multi-array, or explicit subrange indexing where one
semantic range cannot express the walk. Do not add the ndarray offset twice.

## 2. JavaScript wrapper lifetime

`NDArray`, `Mesh`, and other TypeScript wrappers register their embind handles
with `typescript/src/internal/registry.ts`. The `FinalizationRegistry` calls
the embind handle's `.delete()` when the JavaScript wrapper is collected.

There are two distinct operations:

- native `destroy()` releases the shared backing state now;
- embind `.delete()` destroys the C++ wrapper object allocated by embind.

Public `.delete()`/`Symbol.dispose` methods normally call native `destroy()`.
The finalizer can later delete the empty embind wrapper. Shared backing buffers
remain alive while another native handle still owns them.

Do not rely on finalization for bounded resource use. Long-running applications
must be able to dispose large native objects explicitly.

When returning a child handle from a composite object, return an independently
owning shared handle. Deleting the child must not delete the parent, and deleting
the parent must not invalidate a child that still owns the same storage.

## 3. Offset-blocked data

`wasm_offset_blocked_buffer` owns two shared buffers: offsets and packed data.
Its TypeScript wrapper exposes `NDArray` handles over those buffers. This is the
same buffer/range duality as the C++ core:

```text
shared offsets buffer + shared data buffer
-> offset-block range
-> random-access jagged blocks
```

Preserve the offset carrier across the boundary. Do not expand jagged data into
nested JavaScript arrays for native computation, and do not rebuild grouping in
TypeScript when C++ already returned offsets.

## 4. Mesh handles, shared data, and caches

`wasm_mesh<Real>` is a thin `std::shared_ptr<mesh_data<Real>>` handle.
`mesh_data` owns faces, points, transformation, lazy topology/spatial caches,
and source-generation counters.

### Default copy

A normal C++ copy or lambda capture of `wasm_mesh` copies the outer shared
pointer. Both handles refer to the exact same `mesh_data` object.

This is the async lifetime mechanism: capturing a mesh by value keeps its data
alive, and cache construction performed by the worker is visible through the
original JavaScript mesh after `await`.

### `shallow_copy()`

`shallow_copy()` creates a new `mesh_data` object whose member handles initially
share their inner buffers and cache values. The transformation is cleared.
Later `set_faces` or `set_points` reassigns only the new object's slots, advances
its source generations, and causes its dependent cache slots to rebuild.

Thus:

- default copy means shared object identity;
- `shallow_copy()` means a forked slot set with initially shared storage;
- neither means a deep copy of geometry.

Do not replace this with ad-hoc JavaScript cache state. Native generations are
the authority for whether tree, topology, and normal caches are fresh.

## 5. Async dispatcher state machine

The dispatcher is implemented by:

- `typescript/src/internal/AsyncDispatcher.ts`;
- `typescript/cpp/include/trueform/ts/core/async_dispatcher.hpp`;
- `typescript/cpp/src/core/async_dispatcher.cpp`;
- `typescript/cpp/include/trueform/ts/core/promise.hpp`.

Each dispatch allocates a stable `async_context` containing:

- aligned status: `0` pending, `1` complete, `-1` failed;
- a type-erased `std::any` result;
- a type-specific converter from `std::any` to `emscripten::val`.

The context is inserted into a `tbb::concurrent_hash_map` and captured by
`shared_ptr` in the worker. This map is a legitimate rare hash-table use: task
completion and main-thread retrieval are concurrent, and the lookup key is an
opaque context address rather than a bounded domain identity.

The worker stores its result, performs a release-store to status, and issues a
WASM atomic notification. JavaScript waits on an `Int32Array` view of status.
Node receives a ref'd keepalive while `Atomics.waitAsync` is pending because the
wait itself does not keep its event loop alive.

One generic `retrieve(slot)` converts a successful result and erases the task
entry. On failure it erases the entry without attempting result conversion.
Dispatch completion alone does not erase a context, so the JavaScript bridge
must call `retrieve(slot)` on every terminal path. Do not add one retrieval
function per result type.

## 6. Worker boundary

An async worker may use:

- copied scalars and enums;
- `wasm_ndarray`, `wasm_mesh`, and other shared native handles captured by value;
- Trueform buffers, ranges, policies, and algorithms;
- nested oneTBB parallelism used by the core operation.

An async worker must not use:

- `emscripten::val`;
- JavaScript objects, callbacks, or TypedArray methods;
- borrowed references whose owning handle was not captured;
- mutable state concurrently reassigned by JavaScript.

Convert JavaScript values into worker-safe native values before dispatch. Convert
the native result to `emscripten::val` only during `retrieve` on the main JS
thread.

Capture handles by value. Reference capture is a lifetime bug even when the
public TypeScript promise appears to keep its argument in scope.

## 7. Mutation contract during async work

Captured shared ownership prevents deallocation; it does not make mutation
safe. While an async operation is pending, callers must not mutate or reassign
the same mesh/array state used by that operation.

When adding an async wrapper, establish:

1. which native state the worker reads or lazily builds;
2. which handles are captured by value;
3. whether any public setter can race the operation;
4. whether the returned object shares input storage;
5. whether disposal before resolution is safe because the capture owns state.

If the operation cannot satisfy this contract, change the ownership/phase
design. Do not paper over it with JavaScript-side locking.

## 8. Initialization and runtime requirements

`init()` in `typescript/src/native.ts` memoizes one initialization promise,
creates the Emscripten module, installs the dispatcher from exported
`wasmMemory` and `retrieve`, and warms the TBB pool with an async round trip.

The current build enables shared memory, pthreads, and memory growth. Browser
deployment therefore needs the environment required for `SharedArrayBuffer`
and pthread workers, including cross-origin isolation. Custom asset locations
must route both the `.wasm` module and the worker's main script.

Do not create a second module instance or dispatcher casually. Native handles,
status addresses, and memory views belong to the module memory that created
them.

## 9. Adding or changing a binding

Inspect a neighboring operation with the same carrier, dtype matrix, result
shape, and sync/async model. Then verify the owning CMake source list and public
exports.

Required sequence:

1. Validate shapes, dtypes, enums, and optional values once at the boundary.
2. Convert to existing native handles or copy into a `wasm_ndarray` once.
3. Call one C++ implementation for both sync and async paths.
4. For async, capture every input handle/value by value and return `promise_t`.
5. Keep all `emscripten::val` work outside the worker.
6. Move native buffers into result handles; do not element-copy them into JS.
7. Wrap every returned native handle in the TypeScript ownership registry.
8. Expose explicit disposal for state-owning results.
9. Test sync/async equivalence, disposal order, shallow sharing, cache reuse, and
   every registered dtype actually promised by the public API.

Registration layouts vary. Some modules share an implementation header across
float32/float64 translation units; others register together. Current code and
`typescript/CMakeLists.txt` win over remembered convention.

## 10. Review checklist

- Does each TypedArray view have a live native owner?
- Is `.data` treated as borrowed and reacquired after possible memory growth?
- Are zero-copy slices represented by shared storage plus metadata?
- Does default handle copying have the intended shared-object semantics?
- Does `shallow_copy()` fork slots rather than deep-copy geometry?
- Are cache freshness and invalidation native responsibilities?
- Does every async lambda capture owning handles by value?
- Is worker code free of `emscripten::val` and JavaScript access?
- Is result conversion deferred to main-thread `retrieve`?
- Is concurrent mutation forbidden or structurally prevented?
- Are sync and async paths the same computation?
- Are disposal and finalization both safe and independently tested?

## Reference map

- WASM ndarray: `typescript/cpp/include/trueform/ts/core/wasm_ndarray.hpp`
- Offset blocks: `typescript/cpp/include/trueform/ts/core/wasm_offset_blocked_buffer.hpp`
- Mesh handle: `typescript/cpp/include/trueform/ts/core/wasm_mesh.hpp`
- Mesh data/cache generations: `typescript/cpp/include/trueform/ts/core/mesh_data.hpp`
- TypeScript ndarray: `typescript/src/ndarray/NDArray.ts`
- TypeScript mesh: `typescript/src/form/Mesh.ts`
- Finalization: `typescript/src/internal/registry.ts`
- Async bridge: `typescript/src/internal/AsyncDispatcher.ts`
- Native dispatcher: `typescript/cpp/include/trueform/ts/core/async_dispatcher.hpp`
- Initialization: `typescript/src/native.ts`
- Runtime flags and sources: `typescript/CMakeLists.txt`
