# C++ Usage Patterns

How to USE trueform from C++. Every pattern below is from the official documentation.

---

## 1. Primitives

### Creating Primitives

```cpp
// Points
tf::point<float, 3> pt{1.f, 2.f, 3.f};
auto pt = tf::make_point(1.f, 2.f, 3.f);       // deduced: point<float, 3>

// Views from existing data
float buf[3]{1.f, 2.f, 3.f};
auto pview = tf::make_point_view(buf);            // deduces size
auto pview = tf::make_point_view<3>(&buf[0]);     // explicit size

// Vectors
tf::vector<float, 3> v{1.f, 2.f, 3.f};
auto v = tf::make_vector(1.f, 2.f, 3.f);

// Unit vectors (normalizes automatically)
auto uv = tf::make_unit_vector(1.f, 0.f, 0.f);
auto uv = tf::normalized(v);
auto uv = tf::make_unit_vector(tf::unsafe, v);    // skip normalization

// Segments
auto seg = tf::make_segment_between_points(pt0, pt1);
auto seg = tf::make_segment(ids, points);          // indirect via indices
auto [p0, p1] = seg;

// Polygons
auto poly = tf::make_polygon(range_of_points);
auto poly = tf::make_polygon(face_ids, points);    // indirect via indices
auto [id0, id1, id2] = poly.indices();

// Lines and Rays
auto line = tf::make_line(origin, direction);
auto ray = tf::make_ray_between_points(start, end);
auto pt_on_line = line(2.5f);  // parametric evaluation

// Planes
auto plane = tf::make_plane(pt0, pt1, pt2);        // from 3 points
auto plane = tf::make_plane(normal, point);         // from normal + point

// AABB
auto aabb = tf::make_aabb(min_pt, max_pt);
auto aabb = tf::aabb_from(polygons);                // from any geometry

// Conversions
tf::point<double, 3> dpt = pt;                     // implicit float→double
auto fpt = dpt.as<float>();                         // explicit
tf::point<float, 3> zpt = tf::zero;                // zero initialization
```

### Structured Bindings

All primitives support destructuring:
```cpp
auto [x, y, z] = point;
auto [p0, p1] = segment;
auto [v0, v1, v2] = triangle;
```

---

## 2. Ranges (Zero-Copy Views)

### Making Typed Ranges from Flat Data

```cpp
// Points from flat float array
std::vector<float> raw;
auto points = tf::make_points<3>(raw);             // range of point views

// Points from point primitives
std::vector<tf::point<float, 3>> owned;
auto points = tf::make_points(owned);

// External data (zero-copy)
float* coords = get_data();
auto points = tf::make_points<3>(tf::make_range(coords, n * 3));

// Faces from flat index array
std::vector<int> raw_ids;
auto faces = tf::make_faces<3>(raw_ids);            // triangles
auto edges = tf::make_edges(raw_ids);               // edges (blocked by 2)

// Polygons from faces + points
auto polygons = tf::make_polygons(faces, points);
```

### Block and Window Ranges

```cpp
// Fixed blocks
auto triangles = tf::make_blocked_range<3>(raw_ids);
auto [a, b, c] = triangles.front();

// Sliding window (consecutive pairs along a path)
auto pairs = tf::make_slide_range<2>(path_ids);
for (auto [a, b] : pairs) { /* consecutive vertices */ }

// Variable-length blocks
auto blocks = tf::make_offset_block_range(offsets, data);
for (auto block : blocks)
    for (auto value : block) { /* ... */ }
```

### Composition

```cpp
// Indirect (gather by index)
auto selected = tf::make_indirect_range(ids, data);  // data[ids[i]]

// Block indirect (gather + remap within blocks)
auto remapped = tf::make_block_indirect_range(faces, point_map.f());

// Mapped (transform on access)
auto doubled = tf::make_mapped_range(data, [](auto x) { return x * 2; });

// Zip (parallel iteration)
for (auto [pt, normal] : tf::zip(points, normals)) { /* ... */ }

// Enumerate (index + value)
for (auto [i, face] : tf::enumerate(faces)) { /* ... */ }

// Sequence
auto ids = tf::make_sequence_range(n);              // 0, 1, 2, ..., n-1

// Slice / take / drop
auto first_10 = tf::take(range, 10);
auto skip_5 = tf::drop(range, 5);
auto middle = tf::slice(range, 5, 15);
```

### Materialization

```cpp
tf::buffer<int> output;
output.allocate(range.size());
tf::parallel_copy(range, output);
```

---

## 3. Buffers (Owning Storage)

```cpp
// Points
tf::points_buffer<float, 3> pts;
pts.allocate(1000);
pts.emplace_back(1.f, 2.f, 3.f);
auto points_view = pts.points();               // non-owning view

// Polygons (triangles)
tf::polygons_buffer<int, float, 3, 3> mesh;
mesh.faces_buffer().emplace_back(0, 1, 2);
auto polygons = mesh.polygons();               // non-owning view
auto faces = mesh.faces();
auto points = mesh.points();

// Dynamic polygons (mixed sizes)
tf::polygons_buffer<int, float, 3, tf::dynamic_size> mixed;
mixed.faces_buffer().push_back({0, 1, 2});     // triangle
mixed.faces_buffer().push_back({3, 4, 5, 6}); // quad

// Segments
tf::segments_buffer<int, float, 3> segs;
segs.edges_buffer().emplace_back(0, 1);

// Curves
tf::curves_buffer<int, float, 3> curves;
curves.paths_buffer().push_back({0, 1, 2});

// Raw data access
int* face_data = mesh.faces_buffer().data_buffer().data();
float* pt_data = mesh.points_buffer().data_buffer().data();
```

---

## 4. Policy Composition (tag / pipe)

The `|` pipe operator attaches precomputed structures to geometry. This is the core pattern for performance — **pre-tag everything the algorithm needs, so it doesn't rebuild it**.

### Key Principle: Pre-Tagging is Faster

Many algorithms require a tree, face_membership, or manifold_edge_link. If you don't tag them, the algorithm builds them internally (and discards them). If you tag them, the algorithm reuses your precomputed structure.

```cpp
// SLOW: boolean rebuilds tree + fm + mel internally for each call
auto [r1, l1, f1] = tf::make_boolean(poly0, poly1, tf::boolean_op::merge);
auto [r2, l2, f2] = tf::make_boolean(poly0, poly2, tf::boolean_op::merge);
```

### CSG graph: build once, query many (N-ary)

```cpp
// One arrangement of N operands; every expression after is a cheap query.
auto graph = tf::make_csg_graph(forms);   // forms: range of tagged polygons
// optional: sheets range, intersect_config, tf::triangulation_type::refined_cdt

auto diff  = tf::make_csg_mesh(graph, tf::csg::op(0) - tf::csg::op(1));
auto full  = tf::make_csg_mesh(graph);    // full arrangement mesh, no selection
auto [m, tags, faces] = tf::make_csg_mesh(
    graph, tf::csg::op(0) | tf::csg::op(1), tf::return_source_ids);

auto [cells, ids] = tf::make_csg_domains(graph);        // every kept domain
auto seams = tf::make_intersection_curves(graph);        // cross-tag polylines


// FAST: build once, reuse across multiple operations
tf::aabb_tree<int, float, 3> tree0(poly0, tf::config_tree(4, 4));
auto fm0 = tf::make_face_membership(poly0);
auto mel0 = tf::make_manifold_edge_link(poly0);
auto form0 = poly0 | tf::tag(tree0) | tf::tag(fm0) | tf::tag(mel0);

// Now each boolean reuses the precomputed structures
auto [r1, l1, f1] = tf::make_boolean(form0, poly1, tf::boolean_op::merge);
auto [r2, l2, f2] = tf::make_boolean(form0, poly2, tf::boolean_op::merge);
```

**Check the function signature or docs to see exactly which tags a function uses.** Common ones:
- Spatial queries: need `tf::tag(tree)`
- Booleans: need `tf::tag(tree)`, `tf::tag(face_membership)`, `tf::tag(manifold_edge_link)`
- Smoothing/curvature: need `tf::tag(face_membership)` or `tf::tag(vertex_link)`

### Attaching Spatial Trees

```cpp
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto form = polygons | tf::tag(tree);

// Separate point tree (for point-level queries on mesh vertices)
tf::aabb_tree<int, float, 3> pt_tree(polygons.points(), tf::config_tree(4, 4));
auto form = tf::make_polygons(
    polygons.faces(),
    polygons.points() | tf::tag(pt_tree)) | tf::tag(tree);
```

### Attaching Topology

```cpp
auto fm = tf::make_face_membership(polygons);
auto mel = tf::make_manifold_edge_link(polygons);
auto vl = tf::make_vertex_link(polygons);
auto form = polygons | tf::tag(fm) | tf::tag(mel) | tf::tag(tree);
```

### Attaching Transformations

Tagging a transformation means **the geometry is transformed at query time**, not upfront. The original data stays in its local coordinate system. This is how you do collision detection with moving objects — same data, different pose each frame.

```cpp
// Static rotation: data stays local, queries see rotated geometry
auto rotation = tf::make_rotation(tf::deg(45.f), tf::axis<2>);
auto rotated = polygons | tf::tag(rotation);

// Dynamic pose: update the frame, queries see new position
auto frame = tf::make_frame(tf::random_transformation<float, 3>());
auto dynamic = polygons | tf::tag(frame);

// Shared views: same data + tree, different transformation per instance
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto base = polygons | tf::tag(tree);

auto instance_a = base | tf::tag(tf::make_rotation(tf::deg(0.f), tf::axis<2>));
auto instance_b = base | tf::tag(tf::make_rotation(tf::deg(90.f), tf::axis<2>));

// Both share the same tree and data — only the transform differs
auto d = tf::distance(instance_a, instance_b);
```

### Attaching Normals

```cpp
// Per-range normals (range has .normals())
auto pts = tf::make_points<3>(coords) | tf::tag_normals(point_normals);
auto n = pts.normals();

// Per-element zip (each element has .normal())
auto pts = tf::make_points<3>(coords) | tf::zip_normals(point_normals);
auto n = pts.front().normal();

// Tags propagate through transformations
auto rotated_pt = tf::transformed(tagged_point, frame);
auto same_id = rotated_pt.id();            // copied as-is
auto rotated_n = rotated_pt.normal();      // rotated by inverse transpose
```

### Tags are Idempotent

Applying the same tag type twice is a no-op — safe to tag unconditionally:
```cpp
auto tagged = polygon | tf::tag_plane();
auto still_tagged = tagged | tf::tag_plane();  // no-op, no recomputation
```

---

## 5. Spatial Queries

All require tagged forms: `form = polygons | tf::tag(tree)`.

### Distance

```cpp
auto d2 = tf::distance2(form, point);              // squared
auto d = tf::distance(form, segment);               // euclidean
auto d = tf::distance(form0, form1);                // form-to-form
```

### Neighbor Search

```cpp
// Nearest (always finds one)
auto [id, metric_pt] = tf::neighbor_search(form, point);
auto [dist2, closest_pt] = metric_pt;

// Within radius (may not find any)
auto result = tf::neighbor_search(form, point, radius2);
if (result) { auto [id, mp] = result; }

// Form-to-form closest pair
auto [ids, mpp] = tf::neighbor_search(form0, form1);
auto [id0, id1] = ids;

// k-NN
std::array<tf::nearest_neighbor<int, float, 3>, 10> buf;
auto knn = tf::make_nearest_neighbors(buf.begin(), k);
tf::neighbor_search(form, point, knn);
for (auto [id, mp] : knn) { /* sorted by distance */ }
```

### Ray Casting

```cpp
auto config = tf::make_ray_config(0.f, 100.f);

auto result = tf::ray_cast(ray, form, config);
if (result) { auto [prim_id, info] = result; /* info.t */ }

auto hit = tf::ray_hit(ray, form, config);
if (hit) { auto [prim_id, info] = hit; /* info.t, info.point */ }
```

### Intersection Tests

```cpp
bool collide = tf::intersects(form0, form1);
bool hit = tf::intersects(form, polygon);
```

### Gathering IDs

```cpp
// Collect intersecting pairs
std::vector<std::pair<int,int>> pairs;
tf::gather_ids(form0, form1, tf::intersects_f, std::back_inserter(pairs));

// Self-intersection pairs
tf::gather_self_ids(form, tf::intersects_f, std::back_inserter(pairs));

// Within custom predicate
tf::gather_ids(form,
    [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
    [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
    std::back_inserter(ids));
```

### Manual Tree Traversal

```cpp
tf::search(form,
    [&](const auto &bv) { return tf::intersects(bv, query); },
    [&](const auto &prim) { /* process matching primitive */ });

// Dual-tree with thread-safe aggregation
tf::local_vector<std::pair<int,int>> local;
tf::search(form0, form1, tf::intersects_f,
    [&](const auto &p0, const auto &p1) {
        if (tf::intersects(p0, p1))
            local.push_back({p0.id(), p1.id()});
    });
auto result = local.to_vector();
```

---

## 6. Topology

### Connectivity Structures (build-once, query-many)

```cpp
auto fm = tf::make_face_membership(polygons);     // vertex → faces
auto vl = tf::make_vertex_link(polygons);          // vertex → neighbor vertices
auto fl = tf::make_face_link(polygons);            // face → neighbor faces
auto mel = tf::make_manifold_edge_link(polygons);  // face edge → peer face

// Query
for (auto face_id : fm[vertex_id]) { /* ... */ }
for (auto neighbor : vl[vertex_id]) { /* ... */ }
for (auto peer : mel[face_id]) {
    if (peer.is_boundary()) { /* ... */ }
    if (peer.is_simple()) { /* peer.face_peer */ }
}
```

### Mesh Analysis

```cpp
auto bedges = tf::make_boundary_edges(polygons);
auto nmedges = tf::make_non_manifold_edges(polygons);
bool closed = tf::is_closed(polygons);
bool manifold = tf::is_manifold(polygons);
int chi = tf::euler_characteristic(polygons);
```

### Connected Components

```cpp
auto cl = tf::make_manifold_edge_connected_component_labels(polygons);
// cl.n_components, cl.labels (per-face)

auto [components, comp_ids] = tf::split_into_components(polygons, cl.labels);
```

### Boundary Paths

```cpp
auto paths = tf::make_boundary_paths(polygons);
for (auto path : paths)
    for (auto vertex_id : path) { /* ... */ }
```

### Orientation

```cpp
tf::orient_faces_consistently(polygons);           // in-place
tf::ensure_positive_orientation(polygons);          // outward normals
tf::reverse_winding(faces);                         // flip all
```

### Triangulation

```cpp
tf::ear_cutter<int> ec;
ec.build(vertex_ids, points_2d);
for (auto face : ec.faces()) { auto [a, b, c] = face; }
```

---

## 7. Geometry

### Mesh Generation

```cpp
auto box = tf::make_box_mesh<int>(5.f, 2.f, 5.f);
auto sphere = tf::make_sphere_mesh<int>(3.f, 20, 20);
auto cylinder = tf::make_cylinder_mesh<int>(2.f, 10.f, 50);
auto plane = tf::make_plane_mesh<int>(10.f, 10.f, 20, 20);
auto tube = tf::make_tube_mesh<int>(curve, 0.5f, 8);
auto tubes = tf::make_tube_mesh<int>(curves, 0.5f, 8);
```

### Normals and Curvatures

```cpp
auto face_normals = tf::compute_normals(polygons);
auto point_normals = tf::compute_point_normals(polygons);
```

### Registration

```cpp
auto transform = tf::fit_rigid_alignment(source_points, target_points);
auto transform = tf::fit_icp_alignment(source, target, state, config);
auto error = tf::chamfer_error(A, B);
```

### Processing

```cpp
auto triangulated = tf::triangulated(polygons);
auto smoothed = tf::laplacian_smoothed(points, 10, 0.5f);
auto edges = tf::make_sharp_edges(polygons, tf::deg(30.f));
```

---

## 8. Boolean Operations

```cpp
// Basic boolean
auto [mesh, labels, face_labels] = tf::make_boolean(
    poly0, poly1, tf::boolean_op::merge);

// With curves
auto [mesh, labels, face_labels, curves] = tf::make_boolean(
    poly0, poly1, tf::boolean_op::intersection, tf::return_curves);

// Mesh arrangements (N inputs)
const std::array forms{poly0, poly1, poly2};
auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(forms);

// Intersection curves only
auto curves = tf::make_intersection_curves(poly0, poly1);
```

---

## 9. Clean and Reindex

```cpp
auto cleaned = tf::cleaned(polygons);
auto cleaned = tf::cleaned(polygons, tolerance);

auto [components, labels] = tf::split_into_components(polygons, face_labels);
auto merged = tf::concatenated(poly0, poly1, poly2);
auto filtered = tf::reindexed_by_ids(polygons, face_ids);
auto filtered = tf::reindexed_by_mask(polygons, bool_mask);
```

---

## 10. I/O

```cpp
auto mesh = tf::read_stl("model.stl");
auto mesh = tf::read_obj<3>("model.obj");           // triangles
auto mesh = tf::read_obj("model.obj");               // dynamic

tf::write_stl(polygons, "output.stl");
tf::write_obj(polygons, "output.obj");
```

---

## 11. Parallel Algorithms

### Basic Parallel Operations

```cpp
tf::parallel_for_each(range, [](auto&& elem) { /* ... */ });
tf::parallel_for_each(range, func, tf::checked);    // sequential if < 1000
tf::parallel_copy(source, destination);
tf::parallel_fill(buffer, value);
tf::parallel_iota(buffer, 0);
tf::parallel_transform(input, output, func);
```

### Data Generation (Performance-Critical)

These are the workhorses for generating variable-length output in parallel. Use them instead of manual loops + push_back.

**`generic_generate`** — parallel generation into a flat buffer. Each element produces variable-length output. Thread-local buffers are merged automatically.

```cpp
// Generate boundary edges: each face may produce 0-N edges
tf::blocked_buffer<int, 2> boundary_edges;
tf::generic_generate(tf::enumerate(faces), boundary_edges.data_buffer(),
    [&](const auto &pair, auto &buffer) {
        auto [face_id, face] = pair;
        for (int i = 0; i < face.size(); ++i)
            if (is_boundary(face[i], face[(i+1) % face.size()]))
                buffer.push_back(face[i]), buffer.push_back(face[(i+1) % face.size()]);
    });

// With thread-local work buffer (reused across iterations)
tf::generic_generate(input, output,
    tf::small_vector<int, 10>{},  // thread-local scratch
    [&](const auto &elem, auto &out, auto &scratch) {
        scratch.clear();
        process(elem, scratch);
        for (auto v : scratch) out.push_back(v);
    });

// Into multiple buffers simultaneously
tf::generic_generate(input, std::tie(buf_a, buf_b, buf_c),
    [&](const auto &elem, auto &outputs) {
        auto &[a, b, c] = outputs;
        if (cond_a(elem)) a.push_back(...);
        if (cond_b(elem)) b.push_back(...);
    });
```

**`generate_offset_blocks`** — parallel generation of variable-length blocks (like offset_block_buffer). Each input produces one block of variable length.

```cpp
// Build per-vertex face adjacency
tf::offset_block_buffer<int, int> adjacency;
tf::generate_offset_blocks(tf::make_sequence_range(n_verts), adjacency,
    [&](int vertex_id, auto &block) {
        for (auto face_id : compute_neighbors(vertex_id))
            block.push_back(face_id);
    });
```

**`blocked_reduce`** — parallel reduce with thread-local accumulators, merged sequentially. For when you need parallel accumulation into complex structures.

```cpp
tf::buffer<intersection_t> intersections;
tf::buffer<point<float,3>> points;

tf::blocked_reduce(
    tf::enumerate(polygons),
    std::tie(intersections, points),                       // global output
    std::make_tuple(tf::buffer<intersection_t>{},
                    tf::buffer<point<float,3>>{}),         // thread-local accumulators
    [&](const auto &block, auto &local) {                  // parallel: accumulate
        auto &[local_ints, local_pts] = local;
        for (auto [id, poly] : block)
            compute(poly, local_ints, local_pts);
    },
    [&](const auto &local, auto &global) {                 // sequential: merge
        auto &[li, lp] = local;
        auto &[gi, gp] = global;
        gi.reallocate(gi.size() + li.size());
        std::copy(li.begin(), li.end(), gi.begin() + gi.size() - li.size());
        // same for points...
    });
```

**`blocked_reduce_sequenced_aggregate`** — like `blocked_reduce` but guarantees merge order matches input order. Use when output ordering must match input ordering.

---

## 12. Transformations

```cpp
auto T = tf::make_identity_transformation<float, 3>();
auto T = tf::make_transformation_from_translation(vector);
auto R = tf::make_rotation(tf::deg(45.f), tf::axis<2>);
auto R = tf::make_rotation(tf::deg(90.f), axis, pivot);
auto R = tf::make_rotation_aligning(from_dir, to_dir);
auto T = tf::random_transformation<float, 3>();

auto pt_transformed = tf::transformed(point, frame);
auto n_transformed = tf::transformed_normal(normal, frame);
```
