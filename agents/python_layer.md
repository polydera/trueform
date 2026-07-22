# Python/Nanobind Boundary Contract

> **Task-specific authority.** Read this after `working_method.md`,
> `cpp_performance_philosophy.md`, and `cpp_execution_patterns.md` when changing
> Python, nanobind, NumPy ownership, or Python-facing dispatch. This document
> defines boundary invariants. Binding instantiation layouts vary; inspect the
> nearest current feature and every owning CMake manifest before adding files.

The Python layer validates and normalizes Python values, retains NumPy storage,
selects concrete native instantiations, and converts native result buffers back
to NumPy. It does not own an alternate geometry implementation.

## Boundary model

Inputs normally follow this path:

```text
Python facade validates dtype/shape and makes data C-contiguous
-> nanobind wrapper stores ndarray handles by value
-> Trueform ranges point directly into Python-owned NumPy memory
-> synchronous native call performs the computation
```

Native outputs normally follow the opposite ownership path:

```text
Trueform owning buffer
-> `buffer.release()` transfers its allocation
-> nanobind capsule owns the pointer and calls `tf::deallocate<T>`
-> NumPy ndarray uses the capsule as its base owner
```

Inputs are usually zero-copy borrowed storage with retained Python ownership.
Outputs are usually zero-copy ownership transfers from Trueform to NumPy. Do
not collapse those two distinct lifetime models into “NumPy views C++ memory.”

## 1. Input arrays and range construction

Wrappers such as `mesh_data_wrapper`, `point_cloud_data_wrapper`,
`primitive_wrapper`, and `offset_blocked_array_wrapper` store
`nanobind::ndarray` objects by value. Those ndarray handles retain the Python
owner for as long as the native wrapper needs its data.

The C++ wrapper then creates semantic ranges directly over `.data()`:

- points become `points` ranges;
- fixed faces become fixed blocked ranges;
- dynamic faces become offset-block ranges;
- primitive batches retain their primitive type, stride, and batch carrier.

No native algorithm should retain a naked pointer beyond the lifetime of the
wrapper/ndarray handle that owns it.

### Contiguity is a boundary invariant

Much of the native layer interprets an ndarray pointer as flat contiguous data.
The Python facade must therefore normalize accepted arrays before constructing
the wrapper. `Mesh` and `PointCloud` use the validation helpers in
`python/src/trueform/_spatial/_validation.py`; primitive wrappers request
`nanobind::c_contig` where appropriate.

For every new input carrier:

1. validate rank, shape, dtype, and compatible dimensions;
2. make it C-contiguous once at the Python boundary if linear native access is
   required;
3. store the normalized array in the Python facade and/or native wrapper;
4. build ranges over that retained array;
5. test a non-contiguous input when the public API promises to accept one.

Do not silently add inner-loop stride handling to core geometry merely to avoid
one boundary normalization.

## 2. Offset-blocked NumPy data

`OffsetBlockedArray` and `offset_blocked_array_wrapper` preserve the C++
offset-block model as two flat arrays:

```text
offsets: [0, ..., data.size]
data:    packed block contents
```

The native wrapper stores both ndarray handles and constructs an
`offset_block_range` over them. The Python `__getitem__` returns a NumPy slice
view into packed data.

Required invariants at native entry:

- offsets and data use supported integer dtypes;
- their dtypes match when the concrete native wrapper requires that;
- both arrays are one-dimensional and contiguous for linear pointer access;
- the first offset is zero;
- the last offset equals packed data length;
- block order retains its C++ carrier meaning.

One-dimensional shape validation does not imply contiguity. Normalize offsets
and data explicitly in the Python facade before constructing the native wrapper;
the native range walks `.data()` linearly.

Do not convert offset-blocked data to `list[np.ndarray]` before native work.
That loses flat ownership, random-access offsets, and the same implicit join key
used by the core execution pipeline.

## 3. Native result ownership

`python/include/trueform/python/util/make_numpy_array.hpp` is the standard
ownership boundary for native buffers.

For a moved `tf::buffer<T>` it:

1. calls `release()` to obtain the allocation;
2. creates a capsule with `make_capsule`;
3. constructs a shaped NumPy ndarray using that capsule as owner.

`make_capsule` deallocates with `tf::deallocate<T>`, matching Trueform's
allocator. Empty arrays use the shared empty-capsule path rather than inventing
an invalid non-null element.

Overloads preserve structural shape for blocked buffers, points, polygons,
segments, curves, offset blocks, and index maps. Use those overloads rather
than manually reconstructing shapes or copying element-by-element.

A new allocation/copy is appropriate when the requested Python value is not an
owning Trueform buffer, such as materializing a transformation matrix from a
view. Make that ownership transition explicit.

Never expose a NumPy array over a temporary buffer or stack allocation. Never
pair a Trueform allocation with Python's default/free allocator.

## 4. Stateful forms and shared views

`mesh_wrapper` and `point_cloud_wrapper` hold shared native data objects. The
data object retains input ndarrays and lazy caches; transformation state is
local to the outer wrapper.

`shared_view()` therefore means:

- shared geometry arrays and shared cached structures;
- a separate outer wrapper;
- no inherited transformation until the caller sets one.

This is not a deep copy. It is the Python equivalent of multiple transformed
views over one geometry/cache authority.

Mesh setters reassign the stored ndarray and mark dependent caches modified.
In-place writes through a previously exposed NumPy array cannot automatically
signal native cache invalidation. Code that exposes or consumes mutable arrays
must establish an explicit mutation contract; do not assume cache freshness can
be inferred from the pointer.

When adding a cached structure, define:

- which source arrays invalidate it;
- which other caches it depends on;
- who owns its returned NumPy storage;
- whether a shared view observes the same cache slot;
- how reassignment and in-place mutation are handled and tested.

## 5. Python concurrency model

The Python API is synchronous. There is no TypeScript-style promise dispatcher.
A call enters nanobind, performs native work, converts the result, and returns
to Python only after completion.

The binding currently does not install a general `gil_scoped_release` or
nanobind release-GIL call guard. The calling Python thread retains the GIL while
the native operation runs. This does not prevent oneTBB workers from executing
pure C++; it does prevent those workers from using the Python C API.

The rule is strict:

- the calling/GIL-owning thread may create nanobind objects, NumPy arrays,
  capsules, tuples, lists, and exceptions;
- TBB workers may access retained array memory and pure C++ structures;
- TBB workers must not construct or mutate Python/nanobind objects;
- no worker may decref a last Python owner as an incidental side effect.

Do not add `gil_scoped_release` mechanically. First prove that every reachable
native path, lazy cache build, destructor, error path, and callback is free of
Python API access for the entire released interval.

## 6. Compute first, commit second

`python/include/trueform/python/intersect/build_intersect_structures.hpp`
shows the binding-specific phase pattern.

For two or many meshes it:

1. computes trees and missing intersection structures into pure C++ results,
   using `tbb::parallel_invoke` or a `tbb::task_group` across meshes;
2. waits for every native computation;
3. commits results serially on the calling thread by converting buffers to
   NumPy and assigning wrapper cache slots.

`mesh_data_wrapper::compute_face_membership_if_missing` and
`compute_manifold_edge_link_if_missing` return optional native structures.
Their matching `commit_*` methods perform `make_numpy_array` and mutate the
nanobind-owned cache state.

This separation is not ceremony. It is how Python object creation stays off
TBB workers while expensive independent work still runs concurrently.

Use the same shape whenever a Python binding wants parallel precomputation:

```text
prepare retained native views on the calling thread
-> compute pure C++ values in parallel
-> barrier
-> create/assign Python objects on the GIL-owning thread
```

Do not put a mutex around NumPy construction and call it worker-safe.

## 7. Dispatch to concrete native types

Python facades use runtime metadata to select compile-time C++ instantiations.
The common path is:

```text
validate public objects
-> `extract_meta`
-> canonicalize supported operand order
-> build the registered suffix
-> `getattr(_trueform.<module>, name)`
-> pass native wrappers
-> restore public result semantics if operands were swapped
```

`canonicalize_index_order` avoids duplicating symmetric index-type
instantiations. Swapping inputs is not semantically free: boolean labels and
other source-directed outputs must be corrected afterward.

Dispatch naming is an implementation boundary, not a public algorithm. Keep
suffix construction centralized. Do not scatter dtype string concatenation or
duplicate validation in every facade.

Concrete registration layouts differ by module. Some use explicit files for
type combinations, some macros or shared implementations. Inspect the module's
registration source and CMake lists; do not generate a remembered matrix that
the public API does not support.

## 8. Sealed native engines

Long-lived expensive structures such as the Python `CsgGraph` use an opaque
native wrapper rather than exporting their internal ranges.

`csg_graph_wrapper` owns:

- mesh wrappers by value, whose ndarray fields retain NumPy inputs;
- tagged forms built over those wrappers;
- the native `tf::csg_graph` that refers to those forms.

The member order and lifetime chain are structural: wrappers outlive forms,
and forms outlive the graph views built over them. The Python facade may retain
user-facing forms/configuration, but native topology and query state stay in
the sealed engine.

Wide outputs move native buffers through `make_numpy_array`. Python expression
syntax crosses as a compact validated program and is decoded once natively; it
does not reproduce graph evaluation in Python.

Use this pattern when construction is expensive and queries reuse native
structure. Do not expose borrowed internal ranges whose owners Python cannot
keep ordered correctly.

## 9. Adding or changing a binding

Inspect a neighboring operation with the same carriers and dtype matrix.

Required sequence:

1. Keep geometry/topology semantics in the C++ core.
2. Normalize Python inputs once and retain every borrowed ndarray owner.
3. Build semantic ranges over contiguous retained memory.
4. Dispatch to the supported concrete native instantiation.
5. Keep parallel worker phases free of Python/nanobind operations.
6. Move owning native outputs through `make_numpy_array` and its structural
   overloads.
7. Preserve labels, offsets, index maps, and carrier ordering exactly.
8. Export through the current Python package surface.
9. Test supported dtype/dimension/arity combinations, invalid boundary inputs,
   ownership after source deletion, shared-view cache behavior, and
   non-contiguous inputs where supported.

Do not add one file per type combination unless the current module actually
uses that layout. Update all owning manifests for any new source.

## 10. Review checklist

- Does every native pointer have a retained ndarray owner?
- Are all linearly accessed arrays contiguous before native entry?
- Are offsets and packed data preserved rather than expanded into Python lists?
- Does each native output transfer ownership with the matching allocator?
- Can any NumPy array outlive the storage it views?
- Are cache invalidation rules explicit for reassignment and in-place mutation?
- Do shared views intentionally share geometry/cache state?
- Is every TBB worker path free of Python and nanobind operations?
- Are Python object creation and cache commits after a worker barrier?
- Does canonical operand swapping restore directed result semantics?
- Does the registration matrix match the public dtype promise?
- Are sealed engine members ordered so every borrowed range outlives its use?

## Reference map

- Array ownership transfer: `python/include/trueform/python/util/make_numpy_array.hpp`
- Capsule deleter: `python/include/trueform/python/util/make_capsule.hpp`
- Offset blocks: `python/include/trueform/python/core/offset_blocked_array.hpp`
- Python offset facade: `python/src/trueform/_core/offset_blocked_array.py`
- Mesh handle: `python/include/trueform/python/spatial/mesh.hpp`
- Mesh data/caches: `python/include/trueform/python/spatial/mesh_data.hpp`
- Parallel compute/serial commit: `python/include/trueform/python/intersect/build_intersect_structures.hpp`
- Primitive ownership: `python/include/trueform/python/core/primitive_wrapper.hpp`
- Runtime dispatch: `python/src/trueform/_dispatch/`
- Sealed CSG engine: `python/include/trueform/python/csg/csg_graph_impl.hpp`
