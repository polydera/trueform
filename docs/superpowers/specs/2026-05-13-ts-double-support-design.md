# TypeScript Binding: Double-Precision (float64) Support

**Date:** 2026-05-13
**Author:** Žiga Sajovic
**Status:** Approved (brainstorming phase)

## Summary

The trueform TypeScript bindings currently support `float32` coordinates only. This spec adds `float64` (double-precision) as a fully supported dtype across NDArrays, meshes, point clouds, curves, all operations, and I/O. Index types remain `int32` only. Mixed-dtype operations are rejected at the TypeScript boundary.

The architectural shape — `wasm_ndarray<T>` templated over element type, registered under dtype-suffixed Embind names, dispatched in TS via `native()[\`fn_${dtype}\`]` — already exists for `float32`/`int32`/`int8`. This spec extends that shape to include `float64`.

## Goals

1. Add `float64` as a first-class dtype throughout the TS API.
2. Templatize `wasm_mesh`, `wasm_point_cloud`, `wasm_curves` (and their internal caches) over coordinate type `Real`.
3. Instantiate every binding for both `Real = float` and `Real = double`.
4. Reject mixed-dtype inputs at the TS boundary with clear error messages.
5. Add `dtype` option to OBJ I/O. STL I/O stays at the file format's native 32-bit precision.
6. Maintain dispatch ergonomics — user calls remain dtype-implicit (inferred from inputs) except where explicit conversion is needed.

## Non-goals

- Templating index types over int32/int64 (indices stay int32).
- Adding `float16`, `bfloat16`, or arbitrary precision.
- Backwards compatibility with the existing default-float32 behavior for raw `number[]` inputs (this is an intentional breaking change — see Migration).

## API Design

### Dtype identifiers

The runtime dtype tag set extends by one:

```ts
type Dtype = "int8" | "int32" | "float32" | "float64" | "bool";
```

A new TypeScript typed handle joins the existing ones:

```ts
type NDArrayFloat64 = NDArray<Float64Array>;
```

### NDArray factory inference

```ts
tf.ndarray([1, 2, 3])                  // → NDArrayFloat64 (BREAKING; was float32)
tf.ndarray(new Float32Array(...))      // → NDArrayFloat32
tf.ndarray(new Float64Array(...))      // → NDArrayFloat64
tf.ndarray(new Int32Array(...))        // → NDArrayInt32
tf.ndarray(new Int8Array(...))         // → NDArrayInt8

arr.as("float64")                      // explicit conversion
arr.as("float32")
```

JavaScript `number` is internally double-precision. Defaulting `number[]` to `float64` preserves user-provided precision and matches NumPy's default.

### Mesh / PointCloud / Curves

Each carries a `dtype: "float32" | "float64"` property inferred from its points NDArray:

```ts
export function mesh(faces: NDArrayInt32 | Int32Array,
                     points: NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array): Mesh {
  const f = faces instanceof NDArray ? faces : ndarray(faces, [faces.length / 3, 3]);
  const p = points instanceof NDArray ? points : ndarray(points, [points.length / 3, 3]);
  const Native = p.dtype === "float64" ? native().NativeFloat64Mesh : native().NativeFloat32Mesh;
  return new Mesh(Native.create(f._handle, p._handle), p.dtype);
}
```

`Mesh.dtype`, `PointCloud.dtype`, `Curves.dtype` are read-only properties on the wrapping class.

### Operation dispatch

Single-input operations follow the input's dtype:

```ts
distance(mesh, point) → NDArray<mesh.dtype>
```

Multi-input operations validate dtype match at the TS boundary and throw on mismatch:

```ts
if (rays.dtype !== mesh.dtype)
  throw new Error(`dtype mismatch: rays=${rays.dtype}, mesh=${mesh.dtype}`);
```

Dispatch via the existing pattern:

```ts
const raw = native()[`distance_mesh_point_${mesh.dtype}`](mesh._handle, point);
return new NDArray(raw, mesh.dtype);
```

### Output dtype

Output dtype follows input dtype. A boolean of two float64 meshes returns a float64 mesh. There is no cross-dtype output — if inputs disagree, the call throws before reaching C++.

### Operations with no typed input (constructors taking explicit dtype)

`tf.zeros`, `tf.ones`, `tf.full`, `tf.eye`, `tf.arange`, `tf.random` accept `"float64"` as the dtype string. Existing overloads unchanged otherwise. These dispatch on the dtype argument directly — no inference, no validation.

`tf.linspace` does not take a dtype argument. It returns `NDArrayFloat64` by default (was `NDArrayFloat32`).

## I/O

```ts
readStl(buf): Mesh                              // always float32; STL is a 32-bit binary format
readStlData(buf): MeshLike                      // always float32
writeStl(mesh): Uint8Array                      // accepts either dtype; downcasts float64 → 32-bit on write

interface ReadObjOptions {
  dynamic?: boolean;                            // existing
  dtype?: "float32" | "float64";                // new, default "float64"
}

readObj(buf, opts?): Mesh                       // default float64
readObjData(buf, opts?): MeshLike               // same options

writeObj(mesh): string                          // accepts either dtype; text full precision
```

I/O functions follow the same templated registration as every other operation. Nothing special.

## C++ binding layer

### Templates

`wasm_mesh`, `wasm_point_cloud`, `wasm_curves` become class templates over `Real`. Their internal PIMPL types (`mesh_data<Real>`, `point_cloud_data<Real>`) template over `Real` as well. The coordinate-dependent caches inside those PIMPLs — AABB tree (`tf::aabb_tree<int, Real, 3>`) and any cached transformed points — follow `Real`. Index-only caches (face_membership, manifold_edge_link, vertex_link, etc.) stay `int`-only:

```cpp
template <typename Real>
class wasm_mesh {
  std::shared_ptr<mesh_data<Real>> _data;
  // tf::aabb_tree<int, Real, 3>, tf::face_membership<int>, ...
};
```

`wasm_ndarray<T>` is already templated; we add `wasm_ndarray<double>` instantiation.

### Embind names

```
NativeFloat32NDArray, NativeFloat64NDArray, NativeInt32NDArray, NativeInt8NDArray
NativeFloat32Mesh,    NativeFloat64Mesh
NativeFloat32PointCloud, NativeFloat64PointCloud
NativeFloat32Curves,  NativeFloat64Curves
```

Index NDArray names already exist (`NativeInt32NDArray`, `NativeInt8NDArray`) and are unchanged.

### Registration pattern (Approach A)

Every binding `.cpp` defines a templated registration helper and calls it for each dtype:

```cpp
template <typename Real>
auto register_distance_bindings(const std::string &suffix) {
  using namespace emscripten;
  function(("distance_mesh_point_"   + suffix).c_str(), &distance_mesh_point_impl<Real>);
  function(("distance_mesh_segment_" + suffix).c_str(), &distance_mesh_segment_impl<Real>);
  function(("distance_mesh_mesh_"    + suffix).c_str(), &distance_mesh_mesh_impl<Real>);
}

EMSCRIPTEN_BINDINGS(distance) {
  register_distance_bindings<float>("float32");
  register_distance_bindings<double>("float64");
}
```

The implementation functions become templates over `Real`. Most are thin wrappers over trueform's C++ core (which is already generic over coordinate type), so the change is mechanical: replace `float` with `Real` and add the template prefix.

### Async dispatcher

`async_dispatcher` is dtype-agnostic infrastructure. The async wrappers around each operation get registered with the same `_float32` / `_float64` suffixes.

### Internal coordinate-type-dependent details

trueform's boolean / cut / intersect pipelines internally pick an integer type for exact arithmetic based on coordinate type (`float` → `int32` grid, `double` → `int64` grid) via `tf::exact::resolve_int_type`. This selection happens inside trueform's C++ headers — the binding layer just passes `Real` through, and the right integer type is selected automatically.

## TypeScript dispatch infrastructure

The existing dispatch helper:

```ts
function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}
```

stays. The dispatch pattern `native()[\`fn_${nd(dtype)}\`](...)` works unchanged — it just sees a new dtype string `"float64"` and dispatches to the corresponding registered function.

Mixed-dtype validation is centralized in small helpers:

```ts
function assertSameDtype(a: { dtype: string }, b: { dtype: string }, names = ["a", "b"]) {
  if (a.dtype !== b.dtype)
    throw new Error(`dtype mismatch: ${names[0]}=${a.dtype}, ${names[1]}=${b.dtype}`);
}
```

Used at the entry of every multi-input function.

## Tests

The TS test suite is run via:

```
cd typescript
npm run build
node tests/run.mjs
```

This is the verification loop between implementation phases.

**Strategy:**

- For each existing test file (`test_mesh.mjs`, `test_spatial.mjs`, etc.) that exercises float32 operations, parameterize the test bodies over dtype:

  ```ts
  for (const dtype of ["float32", "float64"] as const) {
    test(`distance mesh-point (${dtype})`, () => {
      const points = tf.ndarray(rawPoints).as(dtype);
      // ...
    });
  }
  ```

- A few tests stay single-dtype on purpose: STL I/O test stays float32; OBJ I/O test exercises both dtypes explicitly.

- A new test file `test_dtype_dispatch.mjs` exercises mixed-dtype rejection — every API surface that takes multiple typed inputs gets a "mixed dtype throws" assertion.

## Build size

Templating doubles the size of the templated paths. WASM binary grows; the absolute size is still small. Documented in release notes.

## Migration / breaking changes

This is a major-version breaking release.

1. **`tf.ndarray([1, 2, 3])` now returns `NDArrayFloat64`** (was `NDArrayFloat32`). Meshes built from raw `number[]` are now float64 by default.
2. **`tf.linspace` returns `NDArrayFloat64`** (was `NDArrayFloat32`).
3. **Operations on mixed-dtype inputs throw.** Before, all inputs were float32 implicitly. Now you can construct float64 inputs, and mixing them with float32 inputs throws.

**Migration recipe:** anywhere you want to keep float32 behavior, pass `Float32Array` explicitly or call `.as("float32")` on the constructed NDArray:

```ts
// Before: implicit float32
const mesh = tf.mesh(faces, [0, 0, 0, 1, 0, 0, 0, 1, 0]);

// After (float64 default):
const mesh = tf.mesh(faces, [0, 0, 0, 1, 0, 0, 0, 1, 0]);  // float64

// After (keep float32):
const mesh = tf.mesh(faces, new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
// or:
const mesh = tf.mesh(faces, tf.ndarray([0, 0, 0, 1, 0, 0, 0, 1, 0]).as("float32"));
```

Documented in `RELEASE_NOTES.md` for the version that ships this change.

## Out of scope (for this spec)

- Python bindings (use a different dispatch model, already support both float and double).
- C++ core changes — the core is already generic over coordinate type.
- Any new operations or features beyond dtype dispatch.

## Verification

After each implementation phase, the full verification loop is:

```bash
cd typescript
npm run build && node tests/run.mjs
```

A phase is "done" when this passes. The implementation plan (next document) will break the work into phases with this check between each.
