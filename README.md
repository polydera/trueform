# trueform

**Real-time geometric processing built on composable range-based policies**

`trueform` is a C++ library for real-time geometric processing, built on the principles of composable views and inline policy injection. It operates directly on your *plain-old-data*, by providing semantic views that wrap it with geometric meaning.

From individual primitives to structured ranges, from metadata injection to spatial and topological processing — every operation happens directly on your data; enriched with semantics, without architectural changes.

The library integrates directly at the call site: no boilerplate, no architectural rewrites, no heavyweight setup. It acts as a lightweight, expressive layer over your existing data. Like C++ ranges or lambdas, it lets you build rich, semantic geometry inline, without sacrificing performance or control.

## Key Features

- **Zero-copy views** - Work directly on your data layout with semantic geometric wrappers
- **Composable policies** - Enrich primitives with metadata (id, normal, state) via `tf::tag` and `tf::zip`
- **Spatial acceleration** - `tf::tree` for k-NN, neighbor search, ray casting, and broad-phase queries
- **Topology** - Connectivity structures, path finding, planar embeddings
- **Intersections** - Mesh-mesh curves, self-intersections, scalar field isocontours, planar arrangements
- **Cut operations** - Embed curves as edges, boolean operations
- **Data management** - Efficient cleaning, reindexing, flat buffers
- **Parallel algorithms** - Built on Intel TBB with optimized memory layouts

## From Raw Data to Real-Time Geometry

Here's how trueform enables complex geometric workflows with minimal code—wrapping your existing data, performing queries, and computing results in real time:

```cpp
#include <trueform/trueform.hpp>

// Start with your raw data—no copies, no conversions
std::vector<float> raw_points = {0, 0, 0, 1, 0, 0, 0, 1, 0};
std::vector<int> indices = {0, 1, 2};

auto points = tf::make_points<3>(raw_points);
auto triangles = tf::make_polygons(tf::make_blocked_range<3>(indices), points);
```

**Primitive queries** work directly on geometry:

```cpp
auto triangle = triangles.front();
auto segment = tf::make_segment_between_points(points[0], points[1]);
auto ray = tf::make_ray_between_points(
    tf::point<float, 3>{0.2f, 0.2f, -1.0f},
    tf::point<float, 3>{0.2f, 0.2f, 1.0f});

auto [dist2, pt_on_tri, pt_on_seg] = tf::closest_metric_point_pair(triangle, segment);
bool contains = tf::contains_point(triangle, points[0]);
if (auto hit = tf::ray_hit(ray, triangle)) {
    auto [status, t, hit_point] = hit;
}
```

**Build topology** for mesh-level operations:

```cpp
tf::face_membership<int> face_membership;
face_membership.build(triangles);

tf::manifold_edge_link<int, 3> manifold_edge_link;
manifold_edge_link.build(triangles.faces(), face_membership);

// Label connected components
tf::buffer<int> component_labels;
component_labels.allocate(triangles.size());
auto component_count = tf::label_connected_components<int>(
    component_labels, tf::make_applier(manifold_edge_link));
```

**Spatial acceleration** enables real-time queries on transformed geometry:

```cpp
tf::tree<int, float, 3> tree(triangles, tf::config_tree(4, 4));

auto transform = tf::random_transformation<float, 3>();
auto dynamic_form = tf::make_form(tf::make_frame(transform), tree,
    triangles | tf::tag(face_membership) | tf::tag(manifold_edge_link));
auto static_form = tf::make_form(tree,
    triangles | tf::tag(face_membership) | tf::tag(manifold_edge_link));

// Collision detection
bool does_intersect = tf::intersects(static_form, dynamic_form);
float distance2 = tf::distance2(static_form, dynamic_form);

// Collect all intersecting triangle pairs
std::vector<std::pair<int, int>> collisions;
tf::gather_ids(static_form, dynamic_form, tf::intersects_f,
               std::back_inserter(collisions));
// Compute intersection curves
auto curves = tf::make_intersection_curves(static_form, dynamic_form);
```

**Boolean operations** in real-time:

```cpp
auto [result_mesh, labels, intersection_curves] = tf::make_boolean(
    static_form, dynamic_form, tf::boolean_op::merge, tf::return_curves);
```

**Parallel algorithms** create scalar fields efficiently:

```cpp
auto some_plane = tf::make_plane(triangles.front());
tf::buffer<float> scalars;
scalars.allocate(points.size());
tf::parallel_transform(points, scalars, tf::distance_f(some_plane));
```

**Curve embedding** extracts isocontours directly into the mesh:

```cpp
std::vector<float> cut_values = {0.1f, 0.5f, 1.0f};
auto [contour_mesh, contour_labels, isocontours] = tf::embedded_isocurves<int>(
    triangles, scalars, tf::make_range(cut_values), tf::return_curves);
```

---

This is trueform from a bird's eye view. For comprehensive coverage of all features, patterns, and advanced usage, see the **[complete documentation](https://xlabmedical.github.io/trueform/modules/core)**.

## Installation

**Requirements:**
- C++17 or later
- Intel TBB (Threading Building Blocks)

`trueform` is header-only. Integrate using CMake's `FetchContent`:

```cmake
include(FetchContent)

FetchContent_Declare(
  trueform
  GIT_REPOSITORY https://github.com/xlabmedical/trueform.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(trueform)
target_link_libraries(my_target PRIVATE tf::trueform)
```

## Documentation

Comprehensive documentation is available at **[xlabmedical.github.io/trueform](https://xlabmedical.github.io/trueform)**

- 📚 **[Getting Started](https://xlabmedical.github.io/trueform/getting-started)** - Quick start guide and installation
- 📖 **[Modules](https://xlabmedical.github.io/trueform/modules/core)** - Complete API reference for all modules
- 📊 **[Benchmarks](https://xlabmedical.github.io/trueform/benchmarks)** - Performance comparisons vs VTK, CGAL, nanoflann
- 💡 **[Examples](https://xlabmedical.github.io/trueform/examples)** - Code examples and integration guides
- 📄 **[Publications](https://xlabmedical.github.io/trueform/about/publications)** - Academic papers and research

## License

Trueform is distributed under a dual-license model:
- **Noncommercial use**: PolyForm Noncommercial License 1.0.0
- **Commercial use**: Separate paid agreement with XLAB

See [LICENSE.noncommercial](./LICENSE.noncommercial) and [license documentation](https://xlabmedical.github.io/trueform/about/license) for details. For commercial licensing, contact [ziga.sajovic@xlab.si](mailto:ziga.sajovic@xlab.si).

## Contributing

We welcome contributions! Browse [open issues](https://github.com/xlabmedical/trueform/issues) labeled by difficulty (`easy`, `medium`, `hard`) to find something that matches your experience level.

**Get Started:**
- 📖 Read the full contributing guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 🌐 View on the documentation site: [Contributing Guide](https://xlabmedical.github.io/trueform/about/contributing)

By contributing, you certify that your work may be distributed under both the PolyForm Noncommercial License and any commercial licenses XLAB offers.

## Citation

If you use trueform in your work, please cite:

```bibtex
@software{trueform2025,
    title={trueform: Real-time Geometric Processing},
    author={Sajovic, {\v{Z}}iga and {et al.}},
    year={2025},
    url={https://github.com/xlabmedical/trueform},
    note={Header-only C++ library for real-time geometric processing built on composable range-based policies. Features spatial acceleration,
    topology, intersections, boolean operations, and parallel algorithms.}
}
```
---

**Developed by [XLAB](https://xlab.si)**
