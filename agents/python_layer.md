# Python Layer Analysis

The Python layer exposes trueform via nanobind with numpy interop, dtype-based dispatch, and parametrized type instantiations.

---

## Architecture

```
Python API (tf.boolean_union, tf.Mesh, tf.PointCloud)
    ↓ dispatch via build_suffix()
Python Wrapper (_cut/boolean.py, _spatial/mesh.py)
    ↓ getattr(_trueform.cut, f"boolean_mesh_mesh_{suffix}")
Nanobind C++ Bindings (python/src/cut/boolean_intint33float3d.cpp)
    ↓ calls template implementation
C++ Core (include/trueform/cut/make_boolean.hpp)
```

---

## 1. Nanobind Binding Layer (`python/src/`)

### 1.1 Module Structure

```cpp
NB_MODULE(_trueform, m) {
    tf::py::register_core(m);
    tf::py::register_spatial_module(m);
    tf::py::register_cut(m);
    tf::py::register_geometry_module(m);
    tf::py::register_topology(m);
    tf::py::register_intersect(m);
    tf::py::register_remesh(m);
    tf::py::register_clean(m);
    tf::py::register_reindex(m);
    tf::py::register_io(m);
}
```

Each `register_*` creates a submodule: `_trueform.cut`, `_trueform.spatial`, etc.

### 1.2 Template Instantiation Pattern

C++ templates must be instantiated concretely for Python. Each type combination gets its own `.cpp` file:

```cpp
// python/src/cut/boolean_intint33float3d.cpp
m.def("boolean_mesh_mesh_intint33float3d",
    [](mesh_wrapper<int, float, 3, 3> &mesh0,
       mesh_wrapper<int, float, 3, 3> &mesh1, int op) {
        return boolean(mesh0, mesh1, int_to_boolean_op(op));
    },
    nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));
```

**Naming convention**: `{operation}_{index0}{index1}{ngon0}{ngon1}{real}{dims}d`
- `intint33float3d` = int32×int32, triangles×triangles, float32, 3D
- `intint3dynfloat3d` = int32×int32, triangles×dynamic, float32, 3D
- `int64int6433double3d` = int64×int64, tri×tri, double, 3D

Each file registers overloads for the relevant type combinations (fixed/dynamic ngon variants, with/without curves, etc.).

### 1.3 Wrapper Types

**`mesh_wrapper<Index, RealT, Ngon, Dims>`** (`python/include/trueform/python/spatial/mesh.hpp`):
- Holds `shared_ptr<mesh_data_wrapper>` with numpy-backed faces + points
- Lazy construction: `build_tree()`, `build_face_membership()`, `build_manifold_edge_link()`
- Access: `.tree()`, `.face_membership()`, `.manifold_edge_link()`
- Transformation support: `.transformation()`, `.set_transformation(mat)`

**`offset_blocked_array_wrapper<IndexT, ValueT>`** (`python/include/trueform/python/core/offset_blocked_array.hpp`):
- Wraps two numpy arrays (offsets + data) as C++ `offset_block_range` view
- Validates: first offset = 0, last offset = data size
- Zero-copy: creates `tf::make_offset_block_range()` view directly into numpy memory

### 1.4 Array Return Pattern

C++ results are moved into numpy arrays with zero-copy ownership transfer:

```cpp
template <typename T>
auto make_numpy_array(tf::buffer<T> &&buffer) {
    T *data = buffer.release();              // Take raw pointer
    auto capsule = make_capsule<T>(data);    // Wrap in PyCapsule for dealloc
    return nanobind::ndarray<nanobind::numpy, T, nanobind::shape<-1>>(
        data, {buffer.size()}, capsule);     // Numpy owns via capsule
}
```

Specialized overloads for `polygons_buffer` (returns faces+points tuple), `curves_buffer` (returns paths+points tuple), `offset_block_buffer` (returns offsets+data tuple).

### 1.5 Boolean Implementation

```cpp
template <typename ...Types>
auto boolean(mesh_wrapper<...> &form_wrapper0, mesh_wrapper<...> &form_wrapper1,
             tf::boolean_op op) {
    // Build tagged forms with tree, face_membership, manifold_edge_link
    auto form0 = form_wrapper0.make_primitive_range()
        | tf::tag(form_wrapper0.manifold_edge_link())
        | tf::tag(form_wrapper0.face_membership())
        | tf::tag(form_wrapper0.tree());

    // Call C++ algorithm
    auto [result_mesh, labels, face_labels] = tf::make_boolean(form0, form1, op);

    // Convert to numpy
    return nanobind::make_tuple(
        make_numpy_array(std::move(result_mesh)),
        make_numpy_array(std::move(labels)),
        make_numpy_array(std::move(face_labels)));
}
```

Handles 4 transformation cases (has0×has1) via `if/else` branching with frame tagging.

---

## 2. Python Wrapper Layer (`python/src/trueform/`)

### 2.1 Dispatch System

The core pattern: extract metadata from inputs → build C++ function name suffix → call via `getattr`.

**Metadata extraction** (`_dispatch/meta.py`):
```python
class InputMeta(NamedTuple):
    index_dtype: Optional[np.dtype]
    real_dtype: np.dtype
    ngon: Optional[str]        # '3', 'dyn', or None
    dims: int
    form_name: Optional[str]   # "mesh", "edge_mesh", "point_cloud"

def extract_meta(data) -> InputMeta:
    if isinstance(data, Mesh):
        return InputMeta(index_dtype=..., real_dtype=..., ngon='dyn' if dynamic else str(ngon), ...)
```

**Suffix building** (`_dispatch/suffix.py`):
```python
def dtype_str(dtype):
    return {np.int32: 'int', np.int64: 'int64', np.float32: 'float', np.float64: 'double'}[dtype]

def build_suffix(meta) -> str:
    # "int3float3d" for Mesh(int32, float32, tri, 3D)
    parts = [dtype_str(idx), str(ngon), dtype_str(real), f"{dims}d"]
    return "".join(filter(None, parts))

def build_suffix_pair(meta0, meta1) -> str:
    # "intint33float3d" for two int32/float32/tri/3D meshes
```

**Index order canonicalization** (`_dispatch/canonicalize.py`):
```python
def canonicalize_index_order(form0, form1):
    # C++ implements: int×int, int×int64, int64×int64
    # Swap int64×int32 → int32×int64, return swap flag
    if idx0 == np.int64 and idx1 == np.int32:
        return form1, form0, True
    return form0, form1, False
```

### 2.2 Mesh Class (`_spatial/mesh.py`)

**Dispatch tables**:
```python
_FIXED_SIZE_WRAPPERS = {
    ("Int", "Float", 3, 2): MeshWrapperIntFloat32D,
    ("Int", "Float", 3, 3): MeshWrapperIntFloat33D,
    ("Int", "Double", 3, 3): MeshWrapperIntDouble33D,
    # ... additional combos for other index/real/ngon/dims types
}
```

**Constructor**:
```python
class Mesh:
    def __init__(self, faces, points, transformation=None):
        # Determine real_type from points.dtype (float32 → "Float", float64 → "Double")
        # Determine index_type from faces.dtype (int32 → "Int", int64 → "Int64")
        # Look up wrapper class from dispatch table
        # Create wrapper: self._wrapper = wrapper_class(faces, points)
```

Supports both fixed-size faces (numpy 2D array) and dynamic-size faces (`OffsetBlockedArray`).

### 2.3 Boolean Wrapper (`_cut/boolean.py`)

```python
def boolean_union(mesh0: Mesh, mesh1: Mesh, return_curves=False):
    return _boolean_impl(mesh0, mesh1, _OP_UNION, return_curves)

def _boolean_impl(mesh0, mesh1, op, return_curves):
    # 1. Validate inputs (both Mesh, both 3D, dtypes match)
    # 2. Canonicalize index order (int64×int32 → swap)
    # 3. Build suffix: "intint33float3d"
    # 4. Call: getattr(_trueform.cut, f"boolean_mesh_mesh_{suffix}")(wrapper0, wrapper1, op)
    # 5. If swapped: flip labels (0↔1)
    # 6. Wrap dynamic results in OffsetBlockedArray
    # 7. Return ((faces, points), labels, face_labels) ± curves
```

### 2.4 OffsetBlockedArray (`_core/offset_blocked_array.py`)

Wraps variable-length blocked data (dynamic polygons, curve paths):
```python
class OffsetBlockedArray:
    def __init__(self, offsets: np.ndarray, data: np.ndarray):
        self._wrapper = OffsetBlockedArrayWrapper(offsets, data)
        self.offsets = offsets
        self.data = data

    def __len__(self): return len(self.offsets) - 1
    def __getitem__(self, i): return self.data[self.offsets[i]:self.offsets[i+1]]
```

### 2.5 Public API (`__init__.py`)

Exports organized by domain:
```python
from ._spatial import Mesh, PointCloud, EdgeMesh
from ._cut import boolean_union, boolean_intersection, boolean_difference, isobands
from ._geometry import normals, triangulated, fit_icp_alignment, chamfer_error
from ._topology import label_connected_components, boundary_edges, boundary_paths
from ._io import read_stl, write_stl, read_obj, write_obj
from ._clean import cleaned
from ._reindex import split_into_components, concatenated
from ._remesh import isotropic_remesh, decimate
```

---

## 3. Testing Patterns

### pytest with parametrize
```python
REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boolean_union(dtype, index_dtype):
    mesh0 = create_sphere(dtype, index_dtype)
    mesh1 = create_sphere(dtype, index_dtype, offset=[1, 0, 0])
    result = tf.boolean_union(mesh0, mesh1)
    assert result[0][0].shape[1] == 3  # faces are triangles
```

### Mesh creator factories
```python
def create_3d_triangle_mesh(index_dtype, real_dtype):
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0,0,0], [1,0,0], [0.5,1,0]], dtype=real_dtype)
    return tf.Mesh(faces, points)

MESH_CREATORS = {
    (3, 'triangle'): create_3d_triangle_mesh,
    (3, 'dynamic'): create_3d_dynamic_mesh,
}
```

### numpy assertion helpers
```python
np.testing.assert_array_equal(faces, expected_faces)
np.testing.assert_allclose(points, expected_points, atol=1e-6)
assert np.isclose(distance, 1.0)
```

---

## 4. Blender Integration

**Docs**: `docs/content/py/4.blender/`

**Plugin**: `python/examples/bpy-plugin/` — Blender add-on with:
- `core.py`: `BlenderMesh` class with dirty-tracking (modifications detected, caches invalidated)
- `tools/boolean.py`: Boolean ops integrated into Blender UI
- `tools/curves.py`: Intersection curve visualization

The Blender integration wraps trueform's Mesh class with automatic conversion from/to Blender mesh data, and caches spatial structures across Blender operations.

---

## 5. Build Pipeline

```
python/CMakeLists.txt
    → nanobind_add_module(_trueform ...)  → .so/.pyd
    → Links against tf::trueform
    → Output: python/build/src/trueform/_trueform.so

pip wheel . -w dist  → trueform-*.whl
pytest python/tests  → run test suite
```

---

## 6. Key Design Principles

1. **Runtime dispatch**: Python resolves C++ overloads at call time via `getattr(_trueform.module, func_name)`
2. **Type safety at boundaries**: Validate dtypes, dimensions, and compatibility before dispatching to C++
3. **Canonical ordering**: int64×int32 automatically swapped to int32×int64 with label fixup
4. **Zero-copy where possible**: numpy arrays are views into C++ buffers (via PyCapsule ownership)
5. **Lazy structure building**: tree, face_membership built on demand, not at construction
6. **Comprehensive parametrized tests**: Every operation tested across all dtype combinations
