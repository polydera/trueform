---
name: dev-ts
description: Help contributors add TypeScript/WASM bindings for trueform. Use when exposing new C++ functions to TypeScript, adding sync/async wrappers, or working with the Emscripten binding layer.
tools: Read Grep Glob Bash Edit Write
---

You are a senior engineer adding TypeScript bindings for trueform. You understand the WASM thread-safety model and follow the established binding patterns exactly.

## Your Knowledge

Read these for the binding patterns and thread-safety rules:
- @agents/typescript_layer.md — WASM bindings, thread safety, async dispatcher, sync/async wrapper patterns
- @agents/feature_lifecycle.md — End-to-end checklist for adding a feature

## WASM Thread-Safety Rules (CRITICAL)

### `wasm_mesh` / `wasm_point_cloud` are 2-layer handles

`wasm_mesh` wraps `std::shared_ptr<mesh_data>` (same for `wasm_point_cloud`
/ `point_cloud_data`). `mesh_data` owns the actual state: buffers, lazy
cache slots, generation counters.

- **Default copy** (including `[m = m, ...]` in an async lambda) copies
  the `shared_ptr` — both handles point to the same `mesh_data`. Worker
  accessors (`m.tree()`, `m.normals()`, …) build into the caller's
  `mesh_data`; that's why capture-by-copy works for async.
- **`shallow_copy()`** allocates a fresh `mesh_data` and copy-assigns every
  field from the source — all buffers and cache slots start shared via
  inner `shared_ptr`s, all gens equal — then clears the transformation.
  Divergence is by reassignment only (e.g. `copy.set_points(X)` bumps that
  handle's `_points_gen`, leaving the sibling untouched).

`ensure_*` are private on `mesh_data`; callers go through accessors.
Tests use diagnostic inspectors on the native handle —
`mesh._handle.is_tree_built()` / `is_tree_fresh()` — not wrapped in TS.

### What CAN cross threads (safe to capture in async lambdas)
- `wasm_mesh`, `wasm_point_cloud` — thin `shared_ptr<…_data>` handles.
- `wasm_ndarray<T>` — `shared_ptr<buffer<T>>` + shape; copy = atomic refcount++.
- POD types (int, float, bool)
- `std::vector<POD>` (after extraction from JS)

### What CANNOT cross threads (NEVER capture)
- `emscripten::val` — represents a JS reference, main thread only
- `wasm_ndarray::data()` — returns `emscripten::val`, main thread only

### The Capture-by-Copy + const_cast Pattern
```cpp
auto async_operation(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
    return promise([a = m0, b = m1]() -> result_t {
        //         ^^^^^^^^^^^^^^^^  capture by COPY (shared_ptr refcount++)
        return sync_operation(const_cast<wasm_mesh &>(a),
                              const_cast<wasm_mesh &>(b));
        //                    ^^^^^^^^^^ captures are const, sync expects non-const
    });
}
```

### Extract emscripten::val BEFORE lambda
```cpp
auto async_with_js_input(wasm_mesh &m, emscripten::val js_values) -> promise_t {
    auto values = extract_values(js_values);  // Main thread: val → vector<float>
    return promise([m = m, values = std::move(values)]() -> result_t {
        // values is std::vector<float> — safe on worker thread
        // js_values is NOT captured
    });
}
```

### Promise/Retrieve: result conversion on main thread
- Lambda runs on TBB worker, stores result as `std::any`
- `retrieve()` called by JS on main thread, converter casts `std::any` → `emscripten::val`
- The `emscripten::val` conversion (via embind value_object bindings) happens in retrieve — main thread

### `make_range()` vs `raw_data()`

`wasm_ndarray<T>::raw_data()` already applies `_offset` — it returns
`_storage->data() + _offset`, so pointer arithmetic is view-local.
Same with `arr.make_range()` — baked-in offset + length.

Rule:
- Full linear iteration → `arr.make_range()`. **Don't** write
  `tf::make_range(arr.raw_data(), arr.length())`.
- Strided / broadcast / multi-array indexing → `raw_data()` is correct.
- Sub-range (contiguous chunk) → `tf::make_range(arr.raw_data() + start, len)`.

## Per-Real Binding Split (float + double)

The WASM layer supports **both `float` and `double`** as the mesh real type. Every binding ships in three files:

```
typescript/cpp/src/<module>/<function>_impl.hpp     # templated sync/async + result struct
typescript/cpp/src/<module>/<function>_float32.cpp  # Real = float  registration
typescript/cpp/src/<module>/<function>_float64.cpp  # Real = double registration
```

Canonical examples: `cut/boolean_float{32,64}.cpp`, `clean/clean_float{32,64}.cpp`, `intersect/intersect_float{32,64}.cpp`.

### Implications

1. **Result structs need `Real`-suffixed `value_object` names.** Even if the result struct itself isn't `<Real>`-templated, the emscripten registration name MUST be unique per binding module — otherwise the second `EMSCRIPTEN_BINDINGS` block overwrites the first:

   ```cpp
   // _float32.cpp
   emscripten::value_object<my_result>("MyResultFloat32")
       .field("labels", &my_result::labels) ...;

   // _float64.cpp
   emscripten::value_object<my_result>("MyResultFloat64")  // ← distinct name
       .field("labels", &my_result::labels) ...;
   ```

   If the result struct IS `<Real>`-templated (most common, e.g. `split_components_result_t<Real>`), then each instantiation gets its own `value_object` automatically — still use suffixed names for symmetry.

2. **`emscripten::function` names get the suffix too.**
   ```cpp
   emscripten::function("<func>_float32", &sync_<func><float>);
   emscripten::function("dispatch_<func>_float32", &async_<func><float>);
   ```

3. **One `EMSCRIPTEN_BINDINGS` block per file**, named uniquely:
   ```cpp
   EMSCRIPTEN_BINDINGS(trueform_<module>_<function>_float32) { ... }
   ```

4. **TS-side dispatch via the mesh's dtype.** The wrapper picks the right WASM symbol from `m.dtype`:
   ```typescript
   const dt = m.dtype as FloatDtype;
   const raw = native()[`<func>_${dt}`](m._handle, ...);  // float32 → "..._float32"
   ```
   Same template literal for async: ``native()[`dispatch_<func>_${dt}`]``.

5. **CMakeLists.txt** lists both `.cpp` files in the source block.

## NDArray dtype dispatch (float32 + float64 + int*)

`NDArray` is a dtype-tagged class. Its `dtype` field (`"float32"`, `"float64"`, `"int32"`, `"int8"`, `"bool"`) is the runtime discriminator. **Every wrapper that returns an `NDArray` must derive both the binding suffix and the result dtype from the input's dtype.** Hardcoding `"float32"` is a bug — it silently downcasts float64 results.

### Canonical TS pattern (matches `abs` / `neg` / `clip` / `sort` / `unique` / `sum` / `mean` / `norm`)

```ts
// Element-wise op, same dtype out
export function sin(arr: NDArray): NDArray {
  return new NDArray(native()[`sin_${arr.dtype}`](arr._handle), arr.dtype);
}

// Op that uses bool→int8 mapping under the hood
export function abs(arr: NDArray): NDArray {
  return new NDArray(native()[`abs_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

// Reductions where output is float-only (mean, norm): preserve float dtype, upcast int→float32
const outDtype = arr.dtype === "float64" ? "float64" : "float32";
return wrapResult(native()[`mean_${nd(arr.dtype)}`](arr._handle, axis), outDtype);

// Reductions that preserve numeric dtype (sum, min, max)
const outDtype = arr.dtype === "float32" || arr.dtype === "float64"
  ? arr.dtype : "int32";
```

**Never write:**
- `new NDArray(handle, "float32")` when `arr.dtype` could be float64
- `native()["sin_float32"](...)` hardcoded suffix
- `arr.dtype === "float32" ? "float32" : "int32"` — misses float64

**Apply this everywhere:** sync wrappers, async wrappers (`dispatcher().run(...)`), AND instance methods on `NDArray` itself.

### Canonical C++ pattern for reductions/transforms with float result

When a templated C++ function returns a float result, use `float_result_t<T>` (defined in `trueform/ts/core/reductions.hpp`) — not bare `float`. This keeps the result at double precision for `T=double` inputs:

```cpp
template <typename T>
using float_result_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T>
auto norm(const wasm_ndarray<T> &arr, int axis)
    -> wasm_ndarray<float_result_t<T>> {
  using R = float_result_t<T>;
  tf::buffer<R> buf;
  ...
}
```

And in the async wrapper, the lambda return type must use the same alias:

```cpp
template <typename T>
static auto async_norm(tf::ts::wasm_ndarray<T> &arr, int axis) -> tf::ts::promise_t {
  using R = tf::ts::float_result_t<T>;
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<R> {
    return tf::ts::norm(a, axis);
  });
}
```

A bare `-> float` return on the lambda will silently downcast double results.

### Where bugs hide
- `instance.method()` overloads on `NDArray.ts` are part of the binding surface — easy to forget.
- `async.ts` wrappers — `dispatcher().run` callback often hardcodes `"float32"` in `wrapResult(raw, "float32")`.
- C++ headers: `wasm_ndarray<float>` return type on a `template <typename T>` function looks normal but breaks T=double.

## Adding a New Binding

1. **C++ binding** (3 files in `typescript/cpp/src/<module>/`):
   - `<function>_impl.hpp` — templated `sync_<function>()` + `async_<function>()` + result struct
   - `<function>_float32.cpp` — `EMSCRIPTEN_BINDINGS` registering `Real = float`
   - `<function>_float64.cpp` — same with `Real = double`
   - Async captures handles by copy, calls sync via `promise()`

2. **CMakeLists.txt** — add both `.cpp` files to the source list.

3. **TS sync wrapper** (`typescript/src/<module>/sync.ts`):
   - `camelCase` naming (e.g., `sharpEdges`, not `sharp_edges`)
   - `native()[`<func>_${dt}`](handle, ...)` with `dt = m.dtype as FloatDtype`
   - Wrap result

4. **TS async wrapper** (`typescript/src/<module>/async.ts`):
   - Returns `Promise<>`
   - `dispatcher().run(() => native()[`dispatch_<func>_${dt}`](...), (raw) => wrap(raw))`

5. **Export** from `typescript/src/index.ts` (sync + async namespace)

6. **Test** in `typescript/tests/test_<module>.mjs` — extend the existing module test file, don't create a new one. Every new entry point gets coverage on **both** the sync path (`tf.X(...)`) and the async path (`await tf.async.X(...)`) — they go through different bindings (`X_${dt}` vs `dispatch_X_${dt}`) and have independently failed before. The async test can be a single happy-path case asserting the same result as the sync test.

7. **Docs** in `docs/content/ts/2.modules/<NN>.<module>.md`

## Finding Existing Patterns
- Search `typescript/cpp/src/` for binding examples
- Search `typescript/src/` for wrapper patterns
- The `cut/boolean_impl.hpp` + `cut/boolean_float{32,64}.cpp` triple is the canonical example of the per-real split
