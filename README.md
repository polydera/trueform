# trueform

Real-time geometric processing. Easy to use, robust on real-world meshes.

Spatial queries, mesh booleans, isocontours, topology — at interactive speed on million-polygon meshes. Robust on non-manifold flaps, inconsistent geometry, the artifacts that pipelines accumulate. Header-only C++17; works directly on your data with zero-copy views.

**[▶ Try it live](https://trueform.polydera.com/live-examples/boolean)** — Booleans, collision, isobands in your browser.

**[Documentation](https://trueform.polydera.com)** — Primitives, trees, topology, booleans — step by step.

## Installation

```bash
pip install trueform
```

This installs both Python bindings and C++ headers. For CMake integration:

```cmake
find_package(trueform REQUIRED CONFIG)
target_link_libraries(my_target PRIVATE tf::trueform)
```

Then configure and build your project:

```bash
cmake -B build -Dtrueform_ROOT=$(python -m trueform.cmake)
cmake --build build
```

Or use FetchContent directly:

```cmake
include(FetchContent)
FetchContent_Declare(trueform
  GIT_REPOSITORY https://github.com/xlabmedical/trueform.git
  GIT_TAG main)
FetchContent_MakeAvailable(trueform)
target_link_libraries(my_target PRIVATE tf::trueform)
```

→ [Full installation guide](https://trueform.polydera.com/cpp/getting-started/installation)

**VTK Integration:** Bring trueform performance to VTK applications. Filters and functions that integrate with VTK pipelines. → [VTK documentation](https://trueform.polydera.com/cpp/vtk)

## Quick Tour

```cpp
#include <trueform/trueform.hpp>

// Start with your raw data—no copies, no conversions
std::vector<float> raw_points = {0, 0, 0, 1, 0, 0, 0, 1, 0};
std::vector<int> indices = {0, 1, 2};

auto points = tf::make_points<3>(raw_points);
auto triangles = tf::make_polygons(tf::make_blocked_range<3>(indices), points);
// or maybe faces are variable
std::vector<int> offsets = {0, 1};
auto d_polygons = tf::make_polygons(tf::make_offset_block_range(offsets, indices), points);
// or maybe the indices are a curve
auto segments = tf::make_segments(tf::make_slide_range<2>(indices), points);
// or just read a file
auto mesh = tf::read_stl("file.stl");
```

**Primitive queries** work directly on geometry:

```cpp
auto triangle = triangles.front();
auto segment = segments.back();
auto ray = tf::make_ray_between_points(
    tf::make_point(0.2f, 0.2f, -1.0f),
    tf::make_point(0.2f, 0.2f, 1.0f));

auto [dist2, pt_on_tri, pt_on_seg] = tf::closest_metric_point_pair(triangle, segment);
bool contains = tf::contains_point(triangle, points[0]);
if (auto hit = tf::ray_hit(ray, triangle)) {
    auto [status, t, hit_point] = hit;
}
```

**Mesh analysis** reveals structure and defects:

```cpp
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

auto dynamic_form = tf::make_form(
    tf::make_frame(tf::random_transformation<float, 3>()), tree, polygons);
auto static_form = tf::make_form(tree, polygons);

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

Results at 1M polygons:

| Operation | Input | Time | Speedup | Baseline | TrueForm |
|-----------|-------|------|---------|----------|----------|
| Boolean Union | 2 × 1M | 28 ms | **84×** | CGAL `Simple_cartesian<double>` | reduction diagrams, double |
| Mesh–Mesh Curves | 2 × 1M | 7 ms | **233×** | CGAL `Simple_cartesian<double>` | reduction diagrams, double |
| Self-Intersection | 2 × 1M | 173 ms | **32×** | libigl EPECK (GMP/MPFR) | reduction diagrams, double |
| Isocontours | 1M, 16 cuts | 3.8 ms | **38×** | VTK `vtkContourFilter` | reduction diagrams, float |
| Connected Components | 1M | 15 ms | **10×** | CGAL | parallel union-find |
| Boundary Paths | 1M | 12 ms | **11×** | CGAL | Hierholzer's algorithm |
| k-NN Query | 500K | 1.7 µs | **3×** | nanoflann k-d tree | AABB tree |
| Mesh–Mesh Distance | 2 × 1M | 0.2 ms | **2×** | Coal (FCL) `OBBRSS` | OBBRSS tree |
| Principal Curvatures | 1M | 25 ms | **55×** | libigl | parallel k-ring quadric fitting |

→ [Full benchmarks](https://trueform.polydera.com/cpp/benchmarks) — Detailed comparisons with VTK, CGAL, libigl, Coal, FCL, and nanoflann.

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

Browse [open issues](https://github.com/xlabmedical/trueform/issues) labeled by difficulty. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

If you use trueform in your work, please cite:

```bibtex
@software{trueform2025,
    title={trueform: Real-time Geometric Processing},
    author={Sajovic, {\v{Z}}iga and {et al.}},
    year={2025},
    url={https://github.com/xlabmedical/trueform}
}
```

---

**Developed by [XLAB](https://xlab.si)**

