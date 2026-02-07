# trueform

Real-time geometric processing on NumPy arrays. Easy to use, robust on real-world data.

Spatial queries, mesh booleans, isocontours, topology — at interactive speed on million-polygon meshes. Robust to non-manifold flaps, inconsistent winding, and pipeline artifacts. NumPy in, NumPy out.

**[Documentation](https://trueform.polydera.com/py/getting-started)** | **[Live Examples](https://trueform.polydera.com/live-examples/boolean)**

## Installation

```bash
pip install trueform
```

## Quick Tour

```python
import numpy as np
import trueform as tf

# NumPy arrays in
points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
], dtype=np.float32)
faces = np.array([
    [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
], dtype=np.int32)

mesh = tf.Mesh(faces, points)

# Or read from file
mesh = tf.read_stl("model.stl")
```

**Boolean operations:**

```python
(result_faces, result_points), labels = tf.boolean_union(mesh0, mesh1)

# With intersection curves
(result_faces, result_points), labels, (paths, curve_points) = tf.boolean_union(
    mesh0, mesh1, return_curves=True
)
```

**Spatial queries:**

```python
static_mesh = tf.Mesh(faces0, points0)
dynamic_mesh = tf.Mesh(faces1, points1)
dynamic_mesh.transformation = rotation_matrix

does_intersect = tf.intersects(static_mesh, dynamic_mesh)
distance = tf.distance(static_mesh, dynamic_mesh)
(id0, id1), (dist2, pt0, pt1) = tf.neighbor_search(static_mesh, dynamic_mesh)
neighbors = tf.neighbor_search(dynamic_mesh, static_mesh.points[0], k=10)
for idx, dist2, pt in neighbors:
    pass
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

| Operation | Input | Time | Speedup | Baseline |
|-----------|-------|------|---------|----------|
| Boolean Union | 2 × 1M | 28 ms | **84×** | CGAL `Simple_cartesian<double>` |
| Mesh–Mesh Curves | 2 × 1M | 7 ms | **233×** | CGAL `Simple_cartesian<double>` |
| ICP Registration | 1M | 7.7 ms | **93×** | libigl |
| Self-Intersection | 1M | 78 ms | **37×** | libigl EPECK (GMP/MPFR) |
| Isocontours | 1M, 16 cuts | 3.8 ms | **38×** | VTK `vtkContourFilter` |
| Connected Components | 1M | 15 ms | **10×** | CGAL |
| Boundary Paths | 1M | 12 ms | **11×** | CGAL |
| k-NN Query | 500K | 1.7 µs | **3×** | nanoflann k-d tree |
| Mesh–Mesh Distance | 2 × 1M | 0.2 ms | **2×** | Coal (FCL) `OBBRSS` |
| Principal Curvatures | 1M | 25 ms | **55×** | libigl |

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
