# trueform

[![Tests](https://github.com/polydera/trueform/actions/workflows/tests.yml/badge.svg)](https://github.com/polydera/trueform/actions/workflows/tests.yml)
[![Docs](https://github.com/polydera/trueform/actions/workflows/generate-docs.yml/badge.svg)](https://github.com/polydera/trueform/actions/workflows/generate-docs.yml)
[![Build](https://github.com/polydera/trueform/actions/workflows/build-python.yml/badge.svg)](https://github.com/polydera/trueform/actions/workflows/build-python.yml)
[![PyPI](https://img.shields.io/pypi/v/trueform)](https://pypi.org/project/trueform/)

Real-time geometric processing. Easy to use, robust on real-world data.

Spatial queries, mesh booleans, isocontours, topology — at interactive speed on million-polygon meshes. Robust to non-manifold flaps, inconsistent winding, and pipeline artifacts. Header-only C++17; works directly on your data with zero-copy views.

**[▶ Try it live](https://trueform.polydera.com/live-examples/boolean)** — Interactive mesh booleans, collisions, isobands and more. No install needed. 

**[Documentation and Tutorials](https://trueform.polydera.com)** — Primitives, trees, topology, booleans — step by step.

## Installation

`trueform` is a C++17 header-only library. It depends on [oneTBB](https://github.com/oneapi-src/oneTBB).

```bash
pip install trueform
```

The pip package includes C++ headers with cmake and conan integration modules.

**Conan** (handles TBB automatically):
```bash
python -m trueform.conan create
```

**CMake** (requires TBB installed):
```cmake
find_package(trueform REQUIRED CONFIG)
target_link_libraries(my_target PRIVATE tf::trueform)
```

```bash
cmake -B build -Dtrueform_ROOT=$(python -m trueform.cmake)
```

For manual installation without pip (FetchContent, system install, conan from repo), see the [full installation guide](https://trueform.polydera.com/cpp/getting-started/installation).

## Integrations

- **[VTK](https://trueform.polydera.com/cpp/vtk)** — Filters and functions that integrate with VTK pipelines
- **[Python](https://trueform.polydera.com/py/getting-started)** — NumPy in, NumPy out
- **[Blender](https://trueform.polydera.com/py/blender)** — Cached meshes with automatic updates for live preview

## Quick Tour

```cpp
#include <trueform/trueform.hpp>

// Start with your raw data—no copies, no conversions
std::vector<float> raw_points = {0, 0, 0, 1, 0, 0, 0, 1, 0};
std::vector<int> indices = {0, 1, 2};

auto points = tf::make_points<3>(raw_points);
auto faces = tf::make_faces<3>(indices);
auto triangles = tf::make_polygons(faces, points);
// or maybe faces are variable
std::vector<int> offsets = {0, 3};
auto d_faces = tf::make_faces(offsets, indices);
auto d_polygons = tf::make_polygons(d_faces, points);

auto polygons_buffer = tf::read_stl("file.stl");
auto polygons = polygons_buffer.polygons();
```

**Primitive queries** work directly on geometry:

```cpp
auto polygon = polygons.front();
auto segment = tf::make_segment_between_points(points[0], points[1]);
auto ray = tf::make_ray_between_points(
    tf::make_point(0.2f, 0.2f, -1.0f),
    tf::make_point(0.2f, 0.2f, 1.0f));

auto [dist2, pt_on_poly, pt_on_seg] = tf::closest_metric_point_pair(polygon, segment);
bool contains = tf::contains_point(polygon, points[0]);
if (auto hit = tf::ray_hit(ray, polygon)) {
    auto [status, t, hit_point] = hit;
}
```

**Mesh analysis** reveals structure and defects:

```cpp
auto polygons = polygons_buffer.polygons();
// Connected components
auto [n_components, labels] = tf::make_manifold_edge_connected_component_labels(polygons);
auto [components, component_ids] = tf::split_into_components(polygons, labels);

// Vertex neighborhoods
auto v_link = tf::make_vertex_link(polygons);
auto k2_ring = tf::make_k_rings(v_link, 2);
auto neighs = tf::make_neighborhoods(polygons.points() | tf::tag(v_link), 0.5f);

// Principal curvatures and directions
auto [k0, k1, d0, d1] = tf::make_principal_directions(polygons);

// Boundary curves (open edges)
auto boundary_paths = tf::make_boundary_paths(polygons);
auto boundary_curves = tf::make_curves(boundary_paths, polygons.points());

// Non-manifold edges (shared by >2 faces)
auto bad_edges = tf::make_non_manifold_edges(polygons);
auto bad_segments = tf::make_segments(bad_edges, polygons.points());

// Fix inconsistent face winding
tf::orient_faces_consistently(polygons);
```

**Spatial acceleration** enables queries on transformed geometry:

```cpp
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));

auto dynamic_form = polygons | tf::tag(tree)
    | tf::tag(tf::random_transformation<float, 3>());
auto static_form = polygons | tf::tag(tree);

// Collision detection
bool does_intersect = tf::intersects(static_form, dynamic_form);
float distance2 = tf::distance2(static_form, dynamic_form);

// Collect all intersecting primitive pairs
std::vector<std::pair<int, int>> collisions;
tf::gather_ids(static_form, dynamic_form, tf::intersects_f,
               std::back_inserter(collisions));

// Compute intersection curves
auto curves = tf::make_intersection_curves(static_form, dynamic_form);
```

**Boolean operations** combine meshes:

```cpp
auto [result_mesh, labels] = tf::make_boolean(
    polygons0,
    polygons1 | tf::tag(tf::make_rotation(tf::deg(45.f), tf::axis<2>)),
    tf::boolean_op::merge);

// With intersection curves
auto [result, labels, curves] = tf::make_boolean(
    polygons0, polygons1, tf::boolean_op::intersection, tf::return_curves);
```

**Scalar fields and isocontours**:

```cpp
// Compute distance field from a plane
auto plane = tf::make_plane(polygons.front());
tf::buffer<float> scalars;
scalars.allocate(polygons.points().size());
tf::parallel_transform(polygons.points(), scalars, tf::distance_f(plane));

// Extract isocontours embedded into the mesh
std::vector<float> cut_values = {-0.5f, 0.0f, 0.5f};
auto [contour_mesh, contour_labels, isocontours] = tf::embedded_isocurves(
    polygons, scalars, tf::make_range(cut_values), tf::return_curves);
```

**Mesh cleanup** prepares geometry for processing:

```cpp
// Merge coincident vertices, remove degenerates and duplicates
auto clean_mesh = tf::cleaned(polygons, tf::epsilon<float>);

// Triangulate n-gons
auto tri_mesh = tf::triangulated(polygons);

// Ensure outward-facing normals on closed meshes
tf::ensure_positive_orientation(polygons);
```

→ [Geometry Walkthrough](https://trueform.polydera.com/cpp/examples/mesh-assembly) — A hands-on tour from raw geometry through booleans and connected components.

→ [Modules](https://trueform.polydera.com/cpp/modules) — Primitives, ranges, policies, and the patterns that connect them.

## Benchmarks

Sample comparisons against VTK, CGAL, libigl, Coal, FCL, and nanoflann:

| Operation | Input | Time | Speedup | Baseline | TrueForm |
|-----------|-------|------|---------|----------|----------|
| Boolean Union | 2 × 1M | 28 ms | **84×** | CGAL `Simple_cartesian<double>` | reduction diagrams, double |
| Mesh–Mesh Curves | 2 × 1M | 7 ms | **233×** | CGAL `Simple_cartesian<double>` | reduction diagrams, double |
| ICP Registration | 1M | 7.7 ms | **93×** | libigl | AABB tree, random subsampling |
| Self-Intersection | 1M | 78 ms | **37×** | libigl EPECK (GMP/MPFR) | reduction diagrams, double |
| Isocontours | 1M, 16 cuts | 3.8 ms | **38×** | VTK `vtkContourFilter` | reduction diagrams, float |
| Connected Components | 1M | 15 ms | **10×** | CGAL | parallel union-find |
| Boundary Paths | 1M | 12 ms | **11×** | CGAL | Hierholzer's algorithm |
| k-NN Query | 500K | 1.7 µs | **3×** | nanoflann k-d tree | AABB tree |
| Mesh–Mesh Distance | 2 × 1M | 0.2 ms | **2×** | Coal (FCL) `OBBRSS` | OBBRSS tree |
| Principal Curvatures | 1M | 25 ms | **55×** | libigl | parallel k-ring quadric fitting |

Apple M4 Max, 16 threads, Clang `-O3 -march=native`. Full methodology, interactive charts, source code, and datasets in [benchmarks documentation](https://trueform.polydera.com/cpp/benchmarks).

## Documentation

- [Getting Started](https://trueform.polydera.com/cpp/getting-started) — Installation and first steps
- [Modules](https://trueform.polydera.com/cpp/modules) — Primitives, trees, topology, booleans
- [Benchmarks](https://trueform.polydera.com/cpp/benchmarks) — Performance comparisons
- [Examples](https://trueform.polydera.com/cpp/examples) — Workflows and library comparisons
- [Python Bindings](https://trueform.polydera.com/py/getting-started) — Full API for Python
- [Research](https://trueform.polydera.com/cpp/about/research) — Theory, publications, and citation

## License

Dual-licensed:
- **Noncommercial**: [PolyForm Noncommercial License 1.0.0](./LICENSE.noncommercial)
- **Commercial**: Contact [info@polydera.com](mailto:info@polydera.com)

## Contributing

Browse [open issues](https://github.com/polydera/trueform/issues) labeled by difficulty. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

If you use trueform in your work, please cite:

```bibtex
@software{trueform2025,
    title={trueform: Real-time Geometric Processing},
    author={Sajovic, {\v{Z}}iga and {et al.}},
    organization={XLAB d.o.o.},
    year={2025},
    url={https://github.com/polydera/trueform}
}
```

---

**Developed by [XLAB Medical](https://xlab.si)**
