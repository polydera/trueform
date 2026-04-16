# TypeScript Layer Analysis

The TypeScript layer exposes trueform as a WASM module with typed wrappers, sync/async execution paths, zero-copy NDArray views, and lazy topology caching.

---

## Architecture

```
TypeScript Wrappers (Mesh, NDArray, Curves, async namespace)
    ↓ native().<function>(handle, ...)
EMSCRIPTEN_BINDINGS (C++ functions registered as JS globals)
    ↓ direct call or dispatch_* for async
C++ Implementation (sync) or TBB Thread Pool (async)
    ↓ Atomics.waitAsync polls for completion
Promise resolution → unwrap to TS types
```

---

## 1. WASM Binding Layer (`typescript/cpp/`)

### 1.1 EMSCRIPTEN_BINDINGS Pattern

Every binding file follows this structure:

```cpp
EMSCRIPTEN_BINDINGS(trueform_module_name) {
    // Result types as value_object (copied across boundary)
    emscripten::value_object<result_t>("JSResultName")
        .field("mesh", &result_t::mesh)
        .field("labels", &result_t::labels);

    // Sync functions (execute immediately)
    emscripten::function("operation_name", &sync_operation);

    // Async functions (dispatch to TBB pool)
    emscripten::function("dispatch_operation_name", &async_operation);
}
```

### 1.2 Sync vs Async

**Sync**: Direct execution, blocks main WASM thread.
```cpp
auto sync_boolean_union(wasm_mesh &m0, wasm_mesh &m1) -> labeled_cut_result {
    // compute directly, return result
}
```

**Async**: Captures by copy, returns slot address.
```cpp
auto async_boolean_union(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
    return promise([a = m0, b = m1]() -> labeled_cut_result {
        return sync_boolean_union(const_cast<wasm_mesh &>(a),
                                  const_cast<wasm_mesh &>(b));
    });
}
```

Captures are by value (shared_ptr refcount increment). The lambda runs on a TBB worker thread. `promise_t = uintptr_t` (slot address). JS polls via `Atomics.waitAsync`.

### 1.3 WASM Types — the 2-layer `wasm_mesh` / `wasm_point_cloud`

`wasm_mesh` is a thin handle holding `std::shared_ptr<mesh_data>`.
`mesh_data` owns the actual state: `wasm_ndarray` buffers, every lazy
cache slot (`_tree`, `_fm`, `_mel`, `_fl`, `_vl`, `_normals`,
`_point_normals`, `_he`), and generation counters. `ensure_*` are
private; callers go through accessors (`m.tree()`, `m.normals()`, …)
which trigger the build. Same pattern for `wasm_point_cloud` /
`point_cloud_data`.

Two copy semantics:

1. **Default handle copy** (`wasm_mesh b = a;` / `[m = m, ...]` lambda
   capture) copies only the `shared_ptr` — both handles point to the
   **same** `mesh_data`. This is why async works: a worker's
   `ensure_*` lands on the `mesh_data` the JS-side handle holds, so the
   cache survives the await. Contract: the caller must not reassign
   mesh data while an async op against it is pending.

2. **`shallow_copy()`** allocates a new `mesh_data` and copy-assigns
   every field from the source. Buffers, cache slot values, and gens
   all start shared via inner shared_ptrs — the two handles are
   observationally identical — then the transformation is cleared on
   the copy. Divergence happens only via reassignment: `set_points` /
   `set_faces` reassigns that handle's slot and bumps its gens;
   siblings are untouched. Stale caches on the mutated handle rebuild
   into that handle's slot only.

`wasm_ndarray<T>` is single-layer: `shared_ptr<tf::buffer<T>>` + offset
+ length + shape. `from_buffer()` takes ownership, `from_js()` copies
from a JS TypedArray, `shallow_copy()` returns a new wrapper over the
same storage.

Diagnostic inspectors on the native handle — `is_tree_built()`,
`is_tree_fresh()`, and the per-cache equivalents — expose slot state
for tests; not wrapped in TS. `Mesh.buildTree()` forwards to a void
`build_tree()` on the handle (the `tree()` accessor's `const aabb_tree&`
return can't cross embind).

### 1.4 Lifetime & cleanup

Every state-owning field is `shared_ptr`-backed, so defaulted
destructors release everything via refcount. `destroy()` is a
"release-now" escape hatch — on the handle it's just `_data.reset()`.

---

## 2. TypeScript Wrapper Layer (`typescript/src/`)

### 2.1 Initialization (`native.ts`)

```typescript
let _native: any = null;
let _dispatcher: AsyncDispatcher | null = null;

export async function init(): Promise<void> {
    const createModule = (await import("./trueform_wasm.js")).default;
    _native = await createModule();
    _dispatcher = new AsyncDispatcher(_native.wasmMemory, _native.retrieve);
    await _dispatcher.run(() => _native.init_tbb());  // Warm up TBB
}

export function native(): any { return _native; }
export function dispatcher(): AsyncDispatcher { return _dispatcher; }
```

Must call `tf.init()` before any operations. Singleton pattern.

### 2.2 Mesh Class (`form/Mesh.ts`)

```typescript
export class Mesh {
    readonly _handle: NativeMesh;

    constructor(handle: NativeMesh) {
        this._handle = handle;
        registry.register(this, { handle });  // FinalizationRegistry for auto-cleanup
    }

    get faces(): NDArrayInt32 { return new NDArray(this._handle.faces(), "int32"); }
    set faces(f: NDArrayInt32 | Int32Array) { this._handle.set_faces(f._handle); }

    get points(): NDArrayFloat32 { return new NDArray(this._handle.points(), "float32"); }
    set points(p: NDArrayFloat32 | Float32Array) { this._handle.set_points(p._handle); }

    get faceMembership(): OffsetBlockedBuffer { /* lazy, cached in C++ */ }
    get manifoldEdgeLink(): NDArrayInt32 { /* lazy, cached in C++ */ }

    shallowCopy(): Mesh { return new Mesh(this._handle.shallow_copy()); }
    buildTree(): void { this._handle.build_tree(); }
    delete(): void { this._handle.destroy(); }
    [Symbol.dispose](): void { this._handle.destroy(); }
}
```

### 2.3 NDArray (`ndarray/NDArray.ts`)

```typescript
export class NDArray<T = any> {
    readonly _handle: NativeNDArray<T>;
    readonly dtype: string;

    get data(): T { return this._handle.data(); }  // Zero-copy TypedArray view
    get shape(): number[] { return this._handle.shape(); }

    // Zero-copy operations
    row(i: number): NDArray<T> { return new NDArray(this._handle.row(i), this.dtype); }
    slice(start: number, end?: number): NDArray<T> { /* zero-copy */ }

    // Element-wise operations (dynamic dispatch on dtype)
    add(other: NDArray | number): NDArray<T> {
        const nd = nativeDtype(this.dtype);
        if (typeof other === "number")
            return new NDArray(native()[`add_scalar_${nd}`](this._handle, other), this.dtype);
        return new NDArray(native()[`add_${nd}`](this._handle, other._handle), this.dtype);
    }

    // Reductions: sum, min, max, norm, mean
    // Relational: eq, lt, gt, le, ge, ne
    // Logical: not, and, or
    // Indexing: take, booleanIndex, argsort
}
```

Dynamic function dispatch: `native()[`add_${dtype}`]` calls the dtype-specific C++ binding.

### 2.4 Operation Wrappers

**Sync** (`cut/sync.ts`):
```typescript
export function booleanUnion(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanUnion(m0: Mesh, m1: Mesh, opts: { returnCurves: true }): LabeledCutResultWithCurves;
export function booleanUnion(m0: Mesh, m1: Mesh, opts?: any) {
    if (opts?.returnCurves)
        return wrapLabeledWithCurves(native().boolean_union_with_curves(m0._handle, m1._handle));
    return wrapLabeled(native().boolean_union(m0._handle, m1._handle));
}
```

**Async** (`cut/async.ts`):
```typescript
export async function booleanUnion(m0: Mesh, m1: Mesh, opts?: any) {
    if (opts?.returnCurves)
        return dispatcher().run(
            () => native().dispatch_boolean_union_with_curves(m0._handle, m1._handle),
            (raw) => wrapLabeledWithCurves(raw));
    return dispatcher().run(
        () => native().dispatch_boolean_union(m0._handle, m1._handle),
        (raw) => wrapLabeled(raw));
}
```

**Result wrapping**:
```typescript
function wrapLabeled(raw: any): LabeledCutResult {
    return {
        mesh: new Mesh(raw.mesh),
        labels: new NDArray(raw.labels, "int8"),
        faceLabels: new NDArray(raw.faceLabels, "int32"),
    };
}
```

### 2.5 Spatial Queries

Primitive types mapped to integer codes for C++ dispatch:
```typescript
const PRIM_TYPE = { point: 0, segment: 1, triangle: 2, ray: 3, line: 4, plane: 5, aabb: 6, polygon: 7 };
```

Query functions accept any primitive type and pass the code + NDArray data to C++.

### 2.6 Public API (`index.ts`)

Exports both sync (default) and async namespaces:
```typescript
export { booleanUnion, booleanIntersection, booleanDifference } from './cut/sync';
export { neighborSearch, distance, rayCast } from './spatial/sync';
export * as async from './async';  // tf.async.booleanUnion(...)
```

---

## 3. Testing Patterns

```javascript
test("booleanUnion", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanUnion(s0, s1);
    assert(result.mesh.numberOfFaces > 0);
    assert(result.labels.length === result.mesh.numberOfFaces);
    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
});
```

**Key patterns**:
- Manual lifecycle management: every `.delete()` call is explicit
- Async tests use `async () => { const result = await tf.async.booleanUnion(...); }`
- Handle ownership tests verify data survives parent deletion
- Simple assert-based (not Jest), run via custom `runner.mjs`

---

## 4. CRITICAL: WASM Thread-Safety Rules

This section is essential for anyone adding new TS bindings. The WASM runtime has a two-thread model: **main thread** (JS/WASM linear memory) and **TBB worker threads** (C++ thread pool). Certain types CANNOT cross between them.

### 4.1 What CAN Cross Threads

- **`wasm_mesh` / `wasm_point_cloud`** — thin handles over
  `shared_ptr<mesh_data>` / `shared_ptr<point_cloud_data>`. Copy = atomic
  refcount++ on the inner data. **Both handles then share the same
  `mesh_data`** — this is the mechanism that makes async caches land on
  the JS-visible original (see §1.3). Safe.
- **`wasm_ndarray<T>`** — wraps `shared_ptr<tf::buffer<T>>`. Copy is a refcount increment. Safe.
- **POD types** (int, float, bool) — always safe.
- **`std::vector<POD>`** — safe once copied/moved into the lambda.

### 4.2 What CANNOT Cross Threads

- **`emscripten::val`** — represents a JS reference. **NEVER capture in async lambdas.** Only accessible on main thread.
- **`wasm_ndarray::data()`** — returns `emscripten::val` (typed_memory_view). Only callable from main thread.

### 4.3 The Capture-by-Copy + const_cast Pattern

Every async function follows this pattern:

```cpp
auto async_boolean_union(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
    return promise([a = m0, b = m1]() -> labeled_cut_result {
        //         ^^^^^^^^^^^^^^^^  capture by COPY (shared_ptr refcount++)
        return sync_boolean_union(const_cast<wasm_mesh &>(a),
                                  const_cast<wasm_mesh &>(b));
        //                        ^^^^^^^^^^ lambda captures are const, sync expects non-const
    });
}
```

**Why capture by copy**: The originals are on the main thread's stack. The lambda runs on a TBB worker thread. Copying `wasm_mesh` / `wasm_ndarray` just copies shared_ptrs (atomic refcount increment), keeping the underlying data alive. Crucially for `wasm_mesh`, the copy shares the **same** `mesh_data` as the original (see §1.3), so any cache the worker builds via `m.tree()` / `m.normals()` / etc. lands on the `mesh_data` the JS-visible handle holds.

**Why const_cast**: Lambda captures are const by default. The sync function takes `&` (non-const) because it may trigger lazy cache builds through accessors. `const_cast` is safe here because the only thing being mutated is the inner `mesh_data`'s cache slots, and the await contract guarantees the JS caller is not concurrently mutating the same mesh.

### 4.4 The emscripten::val Extraction Pattern

When a binding receives JS values (e.g., cut values as a JS array), extract them on the main thread BEFORE dispatching:

```cpp
auto async_isobands(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                    emscripten::val js_cut_values) -> promise_t {
    // STEP 1: Extract on main thread (emscripten::val → std::vector)
    auto cv = extract_cut_values(js_cut_values);  // Main thread only!

    // STEP 2: Capture only the extracted data
    return promise([m = mesh, s = scalars, cv = std::move(cv)]() -> result_t {
        // cv is now a std::vector<float> — safe on worker thread
        // js_cut_values is NOT captured
    });
}
```

### 4.5 The Promise/Retrieve Mechanism

```
Main thread                          Worker thread
───────────                          ─────────────
dispatch(fn):
  1. Deduce return type R
  2. Create converter: std::any → R → emscripten::val
  3. Allocate async_context {status, result, converter}
  4. Dispatch fn to TBB pool ──────→ 5. fn() runs, stores result in std::any
  6. Return &status to JS            7. Set status=1, atomic_notify
       ↓
  8. JS: Atomics.waitAsync(status)
       ↓ (status becomes 1)
  9. JS calls retrieve(slot)
 10. converter(result) → emscripten::val  ← Conversion happens HERE (main thread)
 11. Return JS object to Promise resolver
```

**Key**: The result struct (e.g., `labeled_cut_result`) is constructed on the worker thread from thread-safe types (wasm_mesh, wasm_ndarray). The conversion to `emscripten::val` (which creates JS objects) happens in `retrieve()` on the **main thread**.

### 4.6 Rules for Adding New Async Bindings

1. **Sync function first**: Write the sync version that does the actual work
2. **Async wrapper**: Capture all inputs by copy, const_cast as needed
3. **Extract emscripten::val BEFORE lambda**: Convert JS arrays to `std::vector<POD>` on main thread
4. **Return C++ structs from lambda**: Members must all be wasm_* types (shared_ptr-based)
5. **Register result types as `value_object`**: So embind knows how to convert to JS on retrieve
6. **Register both sync and dispatch_* functions** in `EMSCRIPTEN_BINDINGS`

---

## 5. Build Pipeline

```
typescript/CMakeLists.txt
    → Emscripten C++ compilation with -lembind
    → Output: trueform.wasm + trueform.js + trueform.d.ts (via --emit-tsd)
    → TypeScript source compiled by tsc
    → Package: @polydera/trueform on npm
```

Key flags: `-sWASM=1 -sMODULARIZE=1 -sEXPORT_ES6=1 -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency'`

---

## 5. Key Design Principles

1. **Shared ownership via shared_ptr** — all WASM types wrap shared_ptr, safe across async boundaries
2. **Zero-copy views** — `.data` returns TypedArray into WASM heap, `.row()` and `.slice()` share buffer
3. **Lazy topology** — generation counters track staleness, rebuild only when inputs change
4. **Dual execution** — every operation available sync (blocking) and async (Promise-based via TBB)
5. **Dynamic dtype dispatch** — `native()[`op_${dtype}`]` selects correct C++ binding at runtime
6. **FinalizationRegistry** — automatic cleanup of WASM objects when TS objects are GC'd
