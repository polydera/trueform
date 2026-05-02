# trueform — NumPy for Geometry

Fast and exact mesh booleans, spatial queries, arrangements, registration, and remeshing on NumPy arrays. NumPy in, NumPy out.

**[Read the article: The STL for Geometry](https://polydera.com/algorithms/the-stl-for-geometry)** — What the STL philosophy of separating algorithms from data looks like when applied to geometry.

**[polydera.com/trueform](https://polydera.com/trueform)** | **[Documentation](https://trueform.polydera.com/py/getting-started)** | **[▶ Try it live](https://trueform.polydera.com/live-examples/boolean)**

## Installation

```bash
pip install trueform
```

## Quick Tour

**Primitives** — typed NumPy arrays, single or batched:

```python
import numpy as np
import trueform as tf

triangle = tf.Triangle(a=[0, 0, 0], b=[1, 0, 0], c=[0, 1, 0])
segment = tf.Segment([[0, 0, 0], [1, 1, 1]])
ray = tf.Ray(origin=[0.2, 0.2, -1], direction=[0, 0, 1])

# Add a leading dimension for batches
pts = tf.Point(np.random.rand(1000, 3).astype(np.float32))
segs = tf.Segment(start=np.random.rand(500, 3), end=np.random.rand(500, 3))
```

**Meshes** are created from NumPy arrays or read from files:

```python
points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
], dtype=np.float32)
faces = np.array([
    [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
], dtype=np.int32)

mesh = tf.Mesh(faces, points)

# Or read from file
mesh = tf.Mesh(*tf.read_stl("model.stl"))
```

**Queries** — same functions for any combination. Batches broadcast:

```python
# Batch × Primitive — distance field to a plane
plane = tf.Plane(normal=[0, 0, 1], offset=0.0)
scalars = tf.distance(pts, plane)                    # shape (1000,)

# Batch × Form — closest point on mesh for each segment
ids, dist2s, closest = tf.neighbor_search(mesh, segs) # 3 arrays, shape (500,)

# Single × Single
dist2, pt_a, pt_b = tf.closest_metric_point_pair(triangle, segment)

if (t := tf.ray_cast(ray, triangle)) is not None:
    hit_point = ray.origin + t * np.array(ray.direction)
```

**Boolean operations:**

```python
(result_faces, result_points), labels, face_labels = tf.boolean_union(mesh0, mesh1)

# With intersection curves
(result_faces, result_points), labels, face_labels, (paths, curve_points) = tf.boolean_union(
    mesh0, mesh1, return_curves=True
)
```

**Remeshing:**

```python
# Decimate to 50%
dec_faces, dec_points = tf.decimated(mesh, 0.5)

# Isotropic remesh to uniform edge lengths
dec_mesh = tf.Mesh(dec_faces, dec_points)
mel = tf.mean_edge_length(dec_mesh)
rem_faces, rem_points = tf.isotropic_remeshed(dec_mesh, mel)
```

→ [Full documentation](https://trueform.polydera.com/py/modules) covers mesh analysis, topology, isocontours, curvature, and more.

## Examples

- **[Guided Examples](https://trueform.polydera.com/py/examples)** — Step-by-step walkthroughs for spatial queries, topology, and booleans
- **[VTK Integration](https://trueform.polydera.com/py/examples/vtk-integration)** — Interactive VTK applications

Run examples locally:

```bash
git clone https://github.com/polydera/trueform.git
cd trueform/python/examples
pip install vtk  # for interactive examples
python vtk/collision.py mesh.stl
```

## Blender Integration

Cached meshes with automatic dirty-tracking for live preview add-ons. See [Blender docs](https://trueform.polydera.com/py/blender).

## Benchmarks

| Operation | Input | Time | Speedup | Baseline | TrueForm |
|-----------|-------|------|---------|----------|----------|
| Boolean Union | 2 × 1M | 28 ms | **6×** | MeshLib (int32 exact + SoS) | exact predicates, canonical topology |
| Mesh–Mesh Curves | 2 × 1M | 7 ms | **233×** | CGAL `Exact_predicates_inexact_constructions_kernel` | exact predicates, canonical topology |
| ICP Registration | 1M | 7.7 ms | **93×** | libigl | AABB tree, random subsampling |
| Self-Intersection | 1M | 78 ms | **37×** | libigl EPECK (GMP/MPFR) | exact predicates, canonical topology |
| Isocontours | 1M, 16 cuts | 3.8 ms | **38×** | VTK `vtkContourFilter` | exact predicates |
| Connected Components | 1M | 15 ms | **10×** | CGAL | parallel union-find |
| Boundary Paths | 1M | 12 ms | **11×** | CGAL | Hierholzer's algorithm |
| k-NN Query | 500K | 1.7 µs | **3×** | nanoflann k-d tree | AABB tree |
| Mesh–Mesh Distance | 2 × 1M | 0.2 ms | **2×** | Coal (FCL) `OBBRSS` | OBBRSS tree |
| Decimation (50%) | 1M | 72 ms | **50×** | CGAL `edge_collapse` | parallel partitioned collapse |
| Principal Curvatures | 1M | 25 ms | **55×** | libigl | parallel k-ring quadric fitting |

Apple M4 Max, 16 threads, Clang `-O3`. [Full methodology](https://trueform.polydera.com/py/benchmarks)

## License

Dual-licensed: [PolyForm Noncommercial 1.0.0](https://github.com/polydera/trueform/blob/main/LICENSE.noncommercial) for noncommercial use, [commercial licenses](mailto:info@polydera.com) available.

## Contributing

See [CONTRIBUTING.md](https://github.com/polydera/trueform/blob/main/CONTRIBUTING.md) and [open issues](https://github.com/polydera/trueform/issues).

## Citation

```bibtex
@software{trueform2025,
    title={trueform: Real-time Geometric Processing},
    author={Sajovic, {\v{Z}}iga and {et al.}},
    organization={XLAB d.o.o.},
    year={2025},
    url={https://github.com/polydera/trueform}
}
```
