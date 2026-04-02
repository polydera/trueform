# trueform

Real-time geometric processing. Easy to use, robust on real-world data.

Arrangements, booleans, registration, remeshing and queries — at interactive speed on million-polygon meshes. Exact and robust to non-manifold flaps and pipeline artifacts. Header-only C++17, parallel via oneTBB.

**[Documentation](https://trueform.polydera.com/cpp/getting-started)** | **[Live Examples](https://trueform.polydera.com/live-examples/boolean)**

## Installation

```
Install-Package polydera.trueform
```

The package depends on `inteltbb.devel.win`, which is pulled automatically.

## Requirements

- C++17 (`/std:c++17` or later)
- MSVC 19.14+

## Quick Start

```cpp
#include <trueform/trueform.hpp>

// Read meshes
auto mesh = tf::read_stl("surface.stl");
auto polygons = mesh.polygons();

// Spatial queries — build once, query many
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto form = polygons | tf::tag(tree);
auto [id, dist2, pt] = tf::neighbor_search(form, tf::make_point(1.f, 2.f, 3.f));

// Boolean union
auto [result, labels, face_labels] = tf::make_boolean(
    polygons0, polygons1, tf::boolean_op::merge);

// Write result
tf::write_stl(result, "output.stl");
```

## Benchmarks

| Operation | Input | Time | Speedup | Baseline |
|-----------|-------|------|---------|----------|
| Boolean Union | 2 x 1M | 28 ms | **84x** | CGAL `Simple_cartesian<double>` |
| Mesh-Mesh Curves | 2 x 1M | 7 ms | **233x** | CGAL `Simple_cartesian<double>` |
| Decimation (50%) | 1M | 72 ms | **50x** | CGAL `edge_collapse` |
| ICP Registration | 1M | 7.7 ms | **93x** | libigl |
| Connected Components | 1M | 15 ms | **10x** | CGAL |

[Full benchmarks](https://trueform.polydera.com/cpp/benchmarks) — methodology, interactive charts, and datasets.

## Documentation

- [Getting Started](https://trueform.polydera.com/cpp/getting-started) — Installation and first steps
- [Modules](https://trueform.polydera.com/cpp/modules) — Primitives, trees, topology, booleans
- [Examples](https://trueform.polydera.com/cpp/examples) — Workflows and library comparisons

Also available for [Python](https://trueform.polydera.com/py/getting-started) and [TypeScript](https://trueform.polydera.com/ts/getting-started).

## License

Commercial license required. Contact [info@polydera.com](mailto:info@polydera.com).
