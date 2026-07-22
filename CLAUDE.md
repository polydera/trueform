# Trueform C++ Style Guide

> **Read first**: `agents/working_method.md` (how work happens here —
> direction-taking, debugging discipline, gates, task division) and
> `agents/cpp_performance_philosophy.md` (the eight guiding laws).
> Patterns: `agents/cpp_engineering_philosophy.md`; the pipeline:
> `agents/cpp_core_architecture.md`. This file is the style dialect
> and the API quick-reference.

## Code Style

- **Naming**: `snake_case` everywhere. Factory functions use `make_` prefix. Template wrappers use `_like` suffix.
- **Private members**: Leading underscore (`_data`, `_id`)
- **Headers**: `#pragma once`, copyright header, standard includes first, namespace closing comments
- **Trailing return types**:
```cpp
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &a, const point_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1>;
```

## Primitives

| Concept | Owning | View | Factory |
|---------|--------|------|---------|
| Point | `point<T, Dims>` | `point_view<T, Dims>` | `make_point()` |
| Vector | `vector<T, Dims>` | `vector_view<T, Dims>` | `make_vector()` |
| Unit Vector | `unit_vector<T, Dims>` | `unit_vector_view<T, Dims>` | `make_unit_vector()` |

```cpp
auto pt = tf::make_point(1.f, 2.f, 3.f);
auto seg = tf::make_segment_between_points(pt0, pt1);
auto plane = tf::make_plane(pt0, pt1, pt2);
auto aabb = tf::aabb_from(points);
```

## Buffers

### Core Buffers
```cpp
tf::buffer<float> data;                    // Basic flat buffer
data.allocate(100);                        // Uninitialized
data.push_back(1.f);
float* raw = data.release();               // Take ownership

tf::blocked_buffer<int, 3> tris;           // Fixed-size blocks
tris.emplace_back(0, 1, 2);
auto [v0, v1, v2] = tris.front();

tf::offset_block_buffer<int, int> polys;   // Variable-size blocks
polys.push_back({0, 1, 2});                // Triangle
polys.push_back({3, 4, 5, 6});             // Quad
```

### Geometric Buffers

| Buffer | Template | Internal Storage |
|--------|----------|------------------|
| `points_buffer` | `<Real, Dims>` | `buffer<Real>` |
| `unit_vectors_buffer` | `<Real, Dims>` | `buffer<Real>` |
| `segments_buffer` | `<Index, Real, Dims>` | edges + points |
| `polygons_buffer` | `<Index, Real, Dims, N>` | faces + points |
| `curves_buffer` | `<Index, Real, Dims>` | paths + points |

```cpp
tf::polygons_buffer<int, float, 3, 3> triangles;           // N=3: blocked_buffer
tf::polygons_buffer<int, float, 3, 4> quads;               // N=4: blocked_buffer
tf::polygons_buffer<int, float, 3, tf::dynamic_size> mixed; // offset_block_buffer

triangles.faces_buffer().emplace_back(0, 1, 2);
triangles.points_buffer().emplace_back(0.f, 0.f, 0.f);
auto view = triangles.polygons();
int* raw = triangles.faces_buffer().data_buffer().data();
```

## Ranges

```cpp
// Blocked ranges
auto faces = tf::make_blocked_range<3>(flat_ids);      // Fixed blocks
auto segments = tf::make_slide_range<2>(path_ids);     // Sliding window
auto blocks = tf::make_offset_block_range(offsets, data); // Variable blocks

// Indirect access
auto indirect = tf::make_indirect_range(indices, values);

// Utilities
tf::enumerate(range)           // (index, element) pairs
tf::zip(r1, r2, r3)            // Combine ranges
tf::make_sequence_range(100)   // 0..99
tf::take(range, 10) / tf::drop(range, 5) / tf::slice(range, 5, 14)
```

## Mesh Construction

```cpp
// From flat data
auto faces = tf::make_faces<3>(flat_ids);              // Triangles
auto edges = tf::make_edges(flat_ids);                 // Pairs
auto faces = tf::make_faces(offsets, data);            // Dynamic

// Combine with points
auto polygons = tf::make_polygons(faces, points);
auto segments = tf::make_segments(edges, points);

// Access
auto [pt0, pt1, pt2] = polygons.front();
auto [id0, id1, id2] = polygons.front().indices();
polygons.faces();   // face indices only
polygons.points();  // vertex positions only
```

## Tagging & Policies

```cpp
// Tag primitives
auto tagged = polygon | tf::tag_plane();    // Computes and caches
auto plane = tf::make_plane(tagged);        // Returns cached

// Tag ranges
auto pts = points | tf::tag_normals(normals);         // Range accessor: pts.normals()
auto pts = points | tf::zip_normals(normals);         // Element accessor: pts.front().normal()

// Compose structures
auto form = polygons | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

// Transformations (lazy, no copy)
auto rotated = polygons | tf::tag(tf::make_rotation(tf::deg(90.f), tf::axis<0>, pivot));
```

## Topology

```cpp
auto fm = tf::make_face_membership(polygons);     // vertex → faces
auto vl = tf::make_vertex_link(polygons);         // vertex → neighbors
auto fl = tf::make_face_link(polygons);           // face → adjacent faces
auto mel = tf::make_manifold_edge_link(polygons); // per-edge info

auto boundaries = tf::make_boundary_paths(polygons);
auto [labels, n] = tf::make_manifold_edge_connected_component_labels(polygons);
```

## Key Operations

```cpp
// Cleaning (merges coincident vertices)
auto clean = tf::cleaned(polygons, tf::epsilon<float>);
auto [clean, face_map, point_map] = tf::cleaned(polygons, tol, tf::return_index_map);

// Concatenation (auto-offsets indices, promotes to dynamic_size if mixed)
auto combined = tf::concatenated(tris.polygons(), quads.polygons());

// Booleans (use instead of concatenate+clean for closed meshes)
auto [result, labels] = tf::make_boolean(mesh1, mesh2, tf::boolean_op::merge);
// Operations: merge, intersection, left_difference, right_difference

// Arrangements: decompose into classified regions; every arrangement
// surface takes tf::arrangement_config = {intersect_config,
// triangulation_type}, implicit from either alone
auto [m, tags, faces] = tf::make_mesh_arrangements(a.polygons(), b.polygons());
auto [m2, t2, f2, curves] = tf::make_mesh_arrangements(
    tf::make_range(forms), tf::return_curves);
auto [ms, fs] = tf::make_polygon_arrangements(soup.polygons()); // self
auto refined = tf::make_mesh_arrangements(
    a.polygons(), b.polygons(), tf::triangulation_type::refined_cdt);

// Intersection curves (region seam scan; no triangulation built)
auto cb = tf::make_intersection_curves(a.polygons(), b.polygons());
auto sc = tf::make_self_intersection_curves(mesh.polygons());

// N-ary / multi-query: build the csg graph once, query many
auto graph = tf::make_csg_graph(forms);   // + sheets, arrangement_config
// arrangement_config = {intersect_config, triangulation_type}, implicit from either
auto m = tf::make_csg_mesh(graph, tf::csg::op(0) - tf::csg::op(1));
auto [cells, ids] = tf::make_csg_domains(graph);

// Orientation
tf::ensure_positive_orientation(polygons);  // Outward normals for closed mesh

// Triangulation
auto tris = tf::triangulated(polygons);

// Split components
auto [components, ids] = tf::split_into_components(polygons, labels);
```

## Parallel Algorithms

```cpp
tf::parallel_for_each(range, [](auto&& elem) { });
tf::parallel_for_each(tf::zip(r1, r2), [](auto pair) { auto [a, b] = pair; });
tf::parallel_transform(input, output, [](auto x) { return x * 2; });
tf::parallel_fill(data, value);
tf::parallel_iota(data, 0);

// Generate variable-length output
tf::generic_generate(tf::enumerate(input), output_buffer,
    [&](const auto& pair, auto& buffer) {
        const auto& [id, elem] = pair;
        buffer.push_back(...);
    });
```

## Spatial Queries

```cpp
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto form = polygons | tf::tag(tree);

tf::distance(form, point);
tf::intersects(form, polygon);
auto [id, metric_pt] = tf::neighbor_search(form, point);
auto result = tf::ray_cast(ray, form, tf::make_ray_config(0.f, 100.f));
```

## I/O

```cpp
auto mesh = tf::read_stl("model.stl");
auto tris = tf::read_obj<3>("model.obj");
tf::write_stl(polygons, "output.stl");
tf::write_stl(polygons | tf::tag(frame), "transformed.stl");
```

## Critical Patterns

1. **Topology requires shared vertices** — always `tf::cleaned()` mesh soup first
2. **Use booleans for closed meshes** — concatenate+clean creates non-manifold edges
3. **Tag transformations for lazy eval** — `polygons | tf::tag(rotation)` doesn't copy data
4. **Buffer access**: `.polygons()` for view, `.faces_buffer().data_buffer().data()` for raw pointer
