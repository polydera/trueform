# Core

At its core, `trueform` is a collection of geometric primitives that view your data, and ranges of these primitives. The primitives and their ranges can be injected with additional semantics
```
      primitive<Dims, Policy0> → ... → primitive<Dims, Policy_n>
                                                      ↓
primitive_range<Dims, Policy0> → ... → primitive_range<Dims, Policy_n>
                                                      ↓
                                              form<Dims, Policy>
                                                ↑           ↑
                                               tree       frame ← transformation
                                                            ↑
                                                  inverse_transformation
```

## Coordinate System Concepts

Before diving into primitives, it's essential to understand how `trueform` handles coordinate types and dimensions:

```cpp
// Extract coordinate type from any policy
using common_t = tf::coordinate_type_t<Policy0, Policy1>;
common_t scalar = 2.0;

// Extract coordinate dimensions
constexpr std::size_t dims = tf::coordinate_dims_v<Policy>;

// These work with any primitive or primitive range
tf::point<float, 3> pt;
static_assert(tf::coordinate_dims_v<decltype(pt)> == 3);
using coord_t = tf::coordinate_type_t<decltype(pt)>; // float
```

These work on primitives and ranges of primitives.

## Primitives

> **primitive** → primitive_range → form

`trueform` provides a set of geometric primitives, including points, vectors, lines, polygons, AABBs, and transformations. Every primitive is a template of the form:
```cpp
primitive<std::size_t(Dims), Policy>
```
The `Policy` parameter is the engine of the library, defining the primitive's behavior (e.g., whether it owns its data or just views it, does it have a normal, does it have state, etc) and enabling extensions.

> **See:** All primitives may be tagged with additional semantics, as we will learn when we learn about [policies](#policies).

### Points and Vectors

These are the most fundamental primitives, provided in three main variations:

| Concept         | General Template                     | Owning Alias                  | View Alias                         |
| --------------- | ------------------------------------ | ----------------------------- | ---------------------------------- |
| **Vector**      | `tf::vector_like<Dims, Policy>`      | `tf::vector<Type, Dims>`      | `tf::vector_view<Type, Dims>`      |
| **Unit Vector** | `tf::unit_vector_like<Dims, Policy>` | `tf::unit_vector<Type, Dims>` | `tf::unit_vector_view<Type, Dims>` |
| **Point**       | `tf::point_like<Dims, Policy>`       | `tf::point<Type, Dims>`       | `tf::point_view<Type, Dims>`       |

Factory functions like `tf::make_vector` and `tf::make_vector_view` create these primitives.

```cpp
tf::vector<float, 3> v0{1.f, 1.f, 1.f};
float buf[3]{2, 2, 2};
auto vview0 = tf::make_vector_view(buf);
auto vview1 = tf::make_vector_view<3>(&buf[0]);

//
tf::unit_vector<float, 3> uv0{v0};
auto uv1 = tf::make_unit_vector(v0);
auto uv2 = tf::make_unit_vector(vview0);

// this will not normalize
auto uv3 = tf::make_unit_vector(tf::unsafe, v0);
auto uv4 = tf::make_unit_vector_view(uv0);

//
tf::point<float, 3> pt0{{4, 4, 4}};
auto pview0 = tf::make_point_view(buf);
auto pview1 = tf::make_point_view<3>(&buf[0]);
```

They all support standard geometric algebra (the appropriate subset of operations `+`, `-`, `*`), conversions (`.as<T>()`), and essential functions like `tf::dot` and `tf::cross`.

> **NOTE:** Points support a narrows subset of vector algebra (i.e. `pt + pt` is a nonsensical operation). If the full set is needed (such as computing a centroid), you may view the point_like, as a vector_like using `pt.as_vector_view()`.

### Line and Ray

Lines and rays are a composite of a point `origin` and a vector `direction`.

| Concept         | General Template                     | Owning Alias                  |
| --------------- | ------------------------------------ | ----------------------------- |
| **Line**        | `tf::line_like<Dims, Policy>`        | `tf::line<Type, Dims>`        |
| **Ray**         | `tf::ray_like<Dims, Policy>`         | `tf::ray<Type, Dims>`         |

There are no view aliases because a call to `make_line_like(point_like, vector_like)` will produce a view, when the inputs are views. The factory `make_line` always returns an owning alias, as does `make_line_between_points`. Furthermore, the `line_like` is convertible to a line. The same holds true for rays.

They additionally support an `operator()(tf::coordinate_type<Policy> t): origin + t * direction`

### Plane

A plane is a composite of a `unit_vector_like<Dims, Policy>` normal and a `tf::coordinate_type<Policy>` d, where `d` is the negative dot product between the normal and a point in the plane.

| Concept          | General Template                      | Owning Alias                   |
| ---------------- | ------------------------------------- | ------------------------------ |
| **plane**        | `tf::plane_like<Dims, Policy>`        | `tf::plane<Type, Dims>`        |

There are no view aliases because a call to `make_plane_like(unit_vector_like, coordinate_type)` will produce a view, when the inputs are views. The factory `make_plane` always returns an owning alias.

```cpp
tf::point<float, 3> pt;
tf::unit_vector<float, 3> normal;
auto plane0 = tf::make_plane(pt, pt, pt);
auto plane1 = tf::make_plane(normal, pt);
auto plane2 = tf::make_plane(normal, -tf::dot(normal, pt));
```

Furthermore, the `plane_like` is convertible to a `tf::plane`

### Segment

A `tf::segment<Dims, Policy>` is a wrapper around a policy that behaves as a range of `tf::point_like`, with compile-time size of `2`. There are several ways to create one:

```cpp
// creates copies of the two points
auto seg0 = tf::make_segment_between_points(pt0, pt1);
auto [seg0_pt0, seg0_pt1] = seg0;

std::array<tf::point<float, 3>, 2> r; // any range of two points
auto seg1 = tf::make_segment(r);
```

Or, if you have edges that index into a larger range of points:

```cpp
std::array<int, 2> ids{0, 1};
auto seg2 = tf::make_segment(ids, r);

// ids are a view, as are the points
auto [id0, id1] = seg2.indices();
auto [seg2_pt0, seg2_pt1] = seg2;
```

### Polygon

A `tf::polygon<Dims, Policy>` is a wrapper around a policy that behaves as a range of `tf::point_like`. When the policy has has a static size, so does the polygon. There are several ways to create one:

```cpp
std::array<tf::point<float, 3>, 3> r;
auto polygon0 = tf::make_polygon(r);
```

Or, if you have faces that index into a larger range of points:

```cpp
std::array<int, 3> ids{0, 1, 2};
auto polygon1 = tf::make_polygon(ids, r);

// ids are a view, as are the points
auto [id0, id1, id2] = polygon1.indices();
auto [pt0, pt1, pt2] = polygon1;
```

> **NOTE**: Polygons use `tf::static_size` internally to determine the static size of the range being passed into it. All `tf` ranges propagate this static information. We still offer an overload `tf::make_polygon<V>` where the user manually supplies this information.

> **See:** [Policies](#policies) to see how to inject normals and planes into polygons.

### AABB

An axis-aligned-bounding-box is a composite of a point_like `min` and a point_like `max`, representing the minimal and maximal corners.

| Concept         | General Template                     | Owning Alias                  |
| --------------- | ------------------------------------ | ----------------------------- |
| **aabb**        | `tf::aabb_like<Dims, Policy>`        | `tf::aabb<Type, Dims>`        |

There are no view aliases because a call to `make_aabb_like(point_like, point_like)` will produce a view, when the inputs are views. The factory `make_aabb` always returns an owning alias. Furthermore, the `aabb_like` is convertible to a `tf::aabb`

It additionally supports `::diagonal()`, `::center()`, and `::size()` methods.

#### AABB of Primitives

To create an `aabb` of a finite primitive:

```cpp
tf::aabb<float, 3> aabb = tf::aabb_from(primitive);
```

### Transformations and Frames

A `transformation` consists of a rotation `R` and a translation `t`. It maps points and vectors according to the table below:

| **Type**       | **Mapping**        |
|----------------|--------------------|
| `point_like`   | `R.dot(pt) + T`    |
| `vector_like`  | `R.dot(vec)`       |

Similarly to points and vectors, transformations may own, or view your data

| Concept            | General Template                      | Owning Alias                    | View Alias                    |
| ------------------ | ------------------------------------- | ------------------------------- |  ------------------------------- |
| **transformation** |`tf::transformation_like<Dims, Policy>`| `tf::transformation<Type, Dims>`| `tf::transformation_view<Dims, Policy>`

A `transformation_like` is convertible to a `transformation`.
```cpp
tf::transformation_view<float, 3> tr_view{float_ptr};
tf::transformation<float, 3> tr{tr_view};
tr.fill(other_float_ptr);
```
the memory layout of a transformation pointer is `Dims` rows of `Dims+1` columns, where the last element of a row a translation of that dimension. The `coordinate type` is again accessible through `tf::coordinate_type<Policy>`.

We also provide two convenience factories: `tf::make_identity_transformation` and `tf::make_transformation_from_translation`.

---

A `frame_like<Dims, Policy>` is a composition of a transformation and its inverse:

| Concept   | General Template             | Owning Alias           |
| --------- | ---------------------------- | ---------------------- |
| **frame** |`tf::frame_like<Dims, Policy>`| `tf::frame<Type, Dims>`|

There are no view aliases because a call to `make_frame_like(transformation_like, transformation_like)` will produce a view, when the inputs are views. The factory `make_frame` always returns an owning alias.

```cpp
tf::frame<float, 3> frame0;
tf::frame<float, 3> frame1{tr_view};
```

Additionally, `tf::frame` uses a safe policy that ensures the inverse is always computed after changes. This means that to alter the transformation in a frame, you must use

```cpp
frame.fill(float_ptr/*, ptr for inverse*/);
frame.set(transformation /*, inverse_transformation*/);
```

The general `tf::frame_like` offers no such guarantees, but is convertible to a `tf::frame`.

### Transforming primitives

Any primitive from the `tf::` namespace can be transformed using `tf::transformed(_this, _by_transformation)`. Transformations can be performed using either `tf::transformation` or `tf::frame`. This includes composing transformations:

```cpp
auto tr0 = tf::random_transformation<float>();
auto tr1 = tf::random_transformation<float>();
auto tr1_dot_tr0 = tf::transformed(tr0, tr1);
```

## Queries of Primitives

`trueform` provides a set of queries that work across all compatible primitive types.

### General queries

All pairs of primitives support the following queries:

| **Query**                 | **Returns**                         |
|---------------------------|-------------------------------------|
|`distance2`                |Squared distance between primitives  |
|`distance`                 |Distance between primitives          |
|`closest_metric_point`     |`tf::metric_point` (on left argument)|
|`closest_metric_point_pair`|`tf::metric_point_pair`              |
|`intersects`               |Do the primitives intersect          |

A `tf::metric_point` is a composite of a `metric` and a `point`, while `tf::metric_point_pair` is a composite of a `metric`,  `point` first and `point` second.

```cpp
auto [dist2, pt_on_poly0, pt_on_poly1] =
    tf::closest_metric_point_pair(polygon0, polygon1);
```

### Point classification queries

A point may be classified (using `tf::classify`) against a plane and polygon, and additionally against a line, ray and segment in 2D. The function returns either a `tf::sidedness`, or a `tf::containment` enum, depending on the context.

### Ray casting

We distinguish two types of ray operations: `ray_cast` (`intersection_status` and `t`) and `ray_hit` (`intersection_status`, `t` and the `tf::point` of the hit).

Both are supported between a ray and all other primitive types.

```cpp
auto seg = tf::make_segment_between_points(tf::random_point<float, 3>(),
                                            tf::random_point<float, 3>());
auto ray = tf::make_ray_between_points(tf::random_point<float, 3>(),
                                        tf::random_point<float, 3>());
// only consider the t \in [1, 2] as valid. The config may be omitted
auto r_cast = tf::ray_cast(ray, seg, tf::make_ray_config(1.f, 2.f));
if (r_cast) {
auto [status, t] = r_cast;
}
auto r_hit = tf::ray_hit(ray, seg, tf::make_ray_config(1.f, 2.f));
if (r_hit) {
auto [status, t, point] = r_hit;
}
```

## Primitive Ranges and Views

> primitive → **primitive_range** → form

The core philosophy of `trueform` is to work with **views of your existing data**. Before we discuss buffers (which are for output and results), let's understand how to create views of your plain data.

`trueform` provides a set of semantic ranges over primitives. Every primitive_range is a template of the form:
```cpp
primitive_range<std::size_t(Dims), Policy>
```
The `Policy` parameter defines the primitive_range's behavior and enables extensions.

### General Range View Adaptors

While `trueform` is not a range-adaptor library, it provides several range adaptors that allow you to work with your existing data layout directly. Our adaptors preserve and propagate static size information, which can be accessed via `tf::static_size`. 
To create a view of your data:
```cpp
auto r0 = tf::make_range(your_container);
// if you know it has n-elements and this
// is not known to tf::static_size
auto r1 = tf::make_range<N>(your_container);
```

#### Blocks and Sliding Windows

```cpp
std::vector<int> raw_ids;
auto triangle_faces0 = tf::make_blocked_range<3>(raw_ids);
auto [t0id0, t0id1, t0id2] = triangle_faces0.front();
// or if the ids are in legacy VTK format (3, a, b, c, 3, e, f, g)
auto triangle_faces1 = tf::make_tag_blocked_range<3>(raw_ids);
auto [t1id0, t1id1, t1id2] = triangle_faces0.front();
//
auto segments0 = tf::make_blocked_range<2>(raw_ids);
auto [s0id0, s0id1] = segments0.front();
// or if the ids are a sequence of curve points
auto segments1 = tf::make_slide_range<2>(raw_ids);
auto [s1id0, s1id1] = segments1.front();
```

> **NOTE:** `tf::make_blocked_range` and other static variants have a defined value_type which enables them to be sortable. In other words, the `value_type` is a static array that holds a copy of the views data.


#### Offset Block Range

For variable-length blocks with offset arrays:

```cpp
tf::buffer<int> offsets = {0, 3, 7, 10}; // Block boundaries
tf::buffer<int> data = {0,1,2, 3,4,5,6, 7,8,9}; // Packed data

auto blocks = tf::make_offset_block_range(offsets, data);
for (auto block : blocks) {
    for (auto value : block) {
        // Process value
    }
}
```

#### Indirect Range

For index-based access to data:

```cpp
tf::buffer<int> indices = {2, 0, 3, 1};
tf::buffer<float> values = {10.0f, 20.0f, 30.0f, 40.0f};

auto indirect_view = tf::make_indirect_range(indices, values);
// indirect_view[0] == values[2] == 30.0f
// indirect_view[1] == values[0] == 10.0f
```

#### Block Indirect Range

For applying index maps to block-structured data inline.

```cpp
tf::buffer<std::array<int, 3>> faces = {{0,1,2}, {3,4,5}, {6,7,8}};
tf::index_map_buffer<int> face_map; // Maps old face IDs to new point IDs
// Fill face_map...

auto remapped_faces = tf::make_block_indirect_range(faces, face_map.f());
```

#### Other Utility Ranges

```cpp
// Sequence range for indices
auto sequence = tf::make_sequence_range(100); // 0, 1, 2, ..., 99

// Enumeration
auto enumerated = tf::enumerate(data_range); // pairs of (index, value)

// Zip ranges together
auto zipped = tf::zip(range1, range2, range3);

// Take/drop operations
auto first_10 = tf::take(data_range, 10);
auto skip_5 = tf::drop(data_range, 5);

// Slice operations
auto slice = tf::slice(data_range, 5, 14); // Elements 5-14

// Constant range (for default values)
auto constant_values = tf::make_constant_range(default_value, count);

// Mapped range (transform on access)
auto mapped = tf::make_mapped_range(data_range, [](auto x) { return x * 2; });
```

### Points and Vectors

These are the most fundamental primitive ranges, provided in three variants:

|Concept     |Template                  |Factory          |
|------------|--------------------------|-----------------|
|vectors     |vectors<Dims, Policy>     |make_vectors     |
|unit vectors|unit_vectors<Dims, Policy>|make_unit_vectors|
|points      |points<Dims, Policy>      |make_points      |

The factories can take flat ranges as inputs:
```cpp
std::vector<float> raw_pts;
auto pts = tf::make_points<3>(raw_pts);
tf::point_view<float, 3> pt = pts.front();
```
Or simply wrap ranges already containing the concerning primitive:
```cpp
std::vector<tf::point<float, 3>> pts0;
auto pts = tf::make_points(pts0);
```

**Working with your existing data:**
```cpp
// Your existing coordinate data
float* coordinate_array = get_external_coords();
int point_count = get_point_count();

// Zero-copy view creation
auto point_view = tf::make_points<3>(tf::make_view<point_count * 3>(coordinate_array));

// Work with individual points
for (auto point : point_view) {
    // point is a tf::point_view<float, 3>
    float x = point[0];
    float y = point[1];
    float z = point[2];
}
```

> **NOTE:** The `points<Dims, Policy>` provide an additional method `.as_vector_view()`, when one needs complete vector algebra over their points.

> **See:** Points and vectors may be tagged with additional semantics (normals, ids, state, topological connectivity, etc), as we will learn when we learn about [policies](#policies).

### Segments

Curves and embedded graphs are modeled by a range of segments:

|Concept      |Template              |Factory          |
|-------------|----------------------|-----------------|
|segments     |segments<Dims, Policy>|make_segments    |

The factory is used in conjunction [General View Adaptors](#general-range-view-adaptors):

```cpp
// from a sequence of edges
auto segments0 = tf::make_segments(tf::make_blocked_range<2>(edge_indices), points);
// from a sequence of points
auto segments1 = tf::make_segments(tf::make_slide_range<2>(point_indices), points);
```

We can use ids to index into a larger array of points:
```cpp
std::vector<int> ids;
auto segments2 = tf::make_segments(tf::make_blocked_range<2>(ids), points);
auto segments3 = tf::make_segments(tf::make_slide_range<2>(ids), points);
```

The segment elements behave [as expected](#segment):
```cpp
auto [s3pt0, s3pt1] = segments3.front();
auto [s3_id0, s3_id1] = segments3.front().indices();
```

We additionally get two methods, `.edges()` and `.points()`:
```cpp
auto [s3_id0_, s3_id1_] = segments3.edges().front();
auto points_range = segments3.points();
```

### Polygons

Meshes are modeled by a range of polygons:

|Concept      |Template              |Factory          |
|-------------|----------------------|-----------------|
|polygons     |polygons<Dims, Policy>|make_polygons    |

The factory is used in conjunction [General View Adaptors](#general-range-view-adaptors):
```cpp
// from a sequence of points
auto polygons0 = tf::make_polygons(tf::make_blocked_range<3>(point_indices), points);
auto [p0pt0, p0pt1, p0pt2] = polygons0.front();
```

We can use ids to index into a larger array of points:
```cpp
std::vector<int> ids;
auto polygons1 = tf::make_polygons(tf::make_blocked_range<3>(ids), points);
// or if the ids are in legacy VTK format (3, a, b, c, 3, e, f, g)
auto polygons2 =
    tf::make_polygons(tf::make_tag_blocked_range<3>(ids), points);
```

The polygons elements behave [as expected](#polygon):
```cpp
auto [p2pt0, p2pt1, p2pt2] = polygons2.front();
auto [p2_id0, p2_id1, p2_id2] = polygons2.front().indices();
```

We additionally get two methods, `.faces()` and `.points()`:
```cpp
auto [p2_id0_, p2_id1_, p2_id2_] = polygons2.faces().front();
auto points_range = polygons2.points();
```

### Edges and Faces

`trueform` provides semantic wrappers for edge and face data:

#### Edges

```cpp
tf::buffer<std::array<int, 2>> edge_data;
auto edges = tf::make_edges(edge_data);
tf::buffer<int> flat_edges;
auto edges = tf::make_edges(tf::make_blocked_range<2>(flat_edges));

// Process edges
for (auto edge : edges) {
    auto [v0, v1] = edge;
    // Process edge vertices
}
```

#### Faces

```cpp
tf::buffer<std::array<int, 3>> face_data;
auto faces = tf::make_faces(face_data);
tf::buffer<int> flat_faces;
auto faces = tf::make_faces(tf::make_blocked_range<3>(flat_faces));

// Process faces
for (auto face : faces) {
    auto [v0, v1, v2] = face;
    // Process face vertices
}
```

### Paths and Curves

`trueform` supports path and curve representations:

#### Paths

```cpp
tf::offset_block_buffer<int, int> path_data;
path_data.push_back({0, 1, 2, 3}); // First path
path_data.push_back({4, 5, 6});    // Second path

auto paths = tf::make_paths(path_data);
for (auto path : paths) {
    for (auto vertex_id : path) {
        // Process path vertices
    }
}
```

#### Curves

Curves combine paths with point data for geometric processing:

```cpp
tf::offset_block_buffer<int, int> path_indices;
tf::points_buffer<float, 3> curve_points;
// Fill data...

auto curves = tf::make_curves(path_indices, curve_points);
for (auto curve : curves) {
    // Access both indices and geometry
    for (auto vertex_id : curve.indices()) {
        // Process vertex indices
    }
    for (auto point : curve) {
        // Process geometric points
    }
}
```

## Data Structures and Buffers

While views work with your existing data, `trueform` provides efficient data structures for managing and outputting geometric data. **All buffers in `trueform` are flat and composed** - you can always access the underlying memory pointers for efficient data transport.

### Core Buffers

#### Basic Buffer

`tf::buffer<T>` is a lightweight alternative to `std::vector` for POD types:

```cpp
tf::buffer<float> coords;
coords.allocate(1000);  // Allocates uninitialized memory
coords.push_back(1.0f);
coords.emplace_back(2.0f);

// ake ownership of underlying memory
std::size_t byte_size = coords.size() * sizeof(float);
float* raw_ptr = coords.release();

// Standard container interface
for (auto value : coords) {
    // Process value
}
```

Key differences from `std::vector`:
- `allocate()` and `reallocate()` provide uninitialized memory
- Only for trivially constructible/destructible types
- Allows you to take ownership of the underlying pointer

#### Blocked Buffer

`tf::blocked_buffer<T, BlockSize>` manages fixed-size blocks of data:

```cpp
tf::blocked_buffer<int, 3> triangle_indices;
triangle_indices.emplace_back(0, 1, 2);
triangle_indices.push_back({3, 4, 5});

// Access blocks
auto triangle = triangle_indices.front(); // Returns array-like object
auto [v0, v1, v2] = triangle;

// Access underlying flat memory
int* raw_data = triangle_indices.data_buffer().data();
std::size_t total_ints = triangle_indices.size() * 3;
```

#### Offset Block Buffer

`tf::offset_block_buffer<Index, T>` handles variable-length blocks:

```cpp
tf::offset_block_buffer<int, int> polygon_indices;
polygon_indices.push_back({0, 1, 2});     // Triangle
polygon_indices.push_back({3, 4, 5, 6}); // Quad

// Iterate over variable-length blocks
for (auto polygon : polygon_indices) {
    for (auto vertex_id : polygon) {
        // Process vertex
    }
}

// Access underlying flat data
auto& offsets = polygon_indices.offsets_buffer();
auto& data = polygon_indices.data_buffer();
```

### Structured Geometric Buffers

All structured buffers provide both semantic access through range views and direct access to underlying flat memory.

#### Points Buffer

`tf::points_buffer<T, Dims>` efficiently stores point collections:

```cpp
tf::points_buffer<float, 3> points;
points.allocate(1000);
points.emplace_back(1.0f, 2.0f, 3.0f);
points.push_back(tf::make_point<3>(4.0f, 5.0f, 6.0f));

// Semantic access through range view
auto first_point = points.front();
// a tf::points view
auto points_view = points.points();

// Direct access to flat memory
float* coordinates = points.data_buffer().data();
std::size_t total_floats = points.size() * 3;
```

#### Vectors Buffer

`tf::vectors_buffer<T, Dims>` manages vector collections:

```cpp
tf::vectors_buffer<float, 3> vectors;
vectors.allocate(500);
vectors.emplace_back(1.0f, 0.0f, 0.0f);

// Semantic access
auto first_vector = vectors.front();
// a tf::vectors view
auto vectors_view = vectors.vectors();

// Direct access to flat memory
float* vector_data = vectors.data_buffer().data();
```

#### Unit Vectors Buffer

`tf::unit_vectors_buffer<T, Dims>` stores normalized vectors:

```cpp
tf::unit_vectors_buffer<float, 3> normals;
normals.allocate(1000);
normals.push_back(tf::make_unit_vector<3>(1.0f, 0.0f, 0.0f));

// Semantic access
auto normals_view = normals.unit_vectors();
auto first_normal = normals_view.front();

// Direct access to flat memory
float* normal_data = normals.data_buffer().data();
```

#### Segments Buffer

`tf::segments_buffer<Index, RealT, Dims>` manages segment collections:

```cpp
tf::segments_buffer<int, float, 3> segments;
segments.edges_buffer().emplace_back(0, 1);
segments.points_buffer().emplace_back(0.0f, 0.0f, 0.0f);
segments.points_buffer().emplace_back(1.0f, 0.0f, 0.0f);

// Semantic access
auto segments_view = segments.segments();  // Equivalent range view
auto first_segment = segments_view.front();
auto [pt0, pt1] = first_segment;

// Direct access to flat memory
int* edge_indices = segments.edges_buffer().data_buffer().data();
float* point_coords = segments.points_buffer().data_buffer().data();
```

#### Polygons Buffer

`tf::polygons_buffer<Index, RealT, Dims, Ngon>` handles polygon meshes:

```cpp
// Static size (triangles)
// faces buffer is a blocked buffer
tf::polygons_buffer<int, float, 3, 3> triangles;
triangles.faces_buffer().emplace_back(0, 1, 2);

// Dynamic size (mixed polygons)
// faces buffer is an offset block buffer
tf::polygons_buffer<int, float, 3, tf::dynamic_size> mixed_polygons;
mixed_polygons.faces_buffer().push_back({0, 1, 2});     // Triangle
mixed_polygons.faces_buffer().push_back({3, 4, 5, 6}); // Quad

// Semantic access
auto polygons_view = mixed_polygons.polygons();  // Equivalent range view
auto mesh_points = mixed_polygons.points();
auto mesh_faces = mixed_polygons.faces();

// Direct access to flat memory
int* face_data = mixed_polygons.faces_buffer().data_buffer().data();
float* point_data = mixed_polygons.points_buffer().data_buffer().data();
```

#### Curves Buffer

`tf::curves_buffer<Index, RealT, Dims>` handles polygon meshes:

```cpp
// Static size (triangles)
tf::curves_buffer<int, float, 3> curves;
curves.paths_buffer().push_back({0, 1, 2});

// Semantic access
auto curves_view = curves.curves();  // Equivalent range view
auto curve_points = curves.points();
auto curve_paths = curves.paths();

// Direct access to flat memory
int* face_data = curves.paths_buffer().data_buffer().data();
float* point_data = curves.points_buffer().data_buffer().data();
```

### Index Maps

`tf::index_map<Range0, Range1>` manages mapping between old and new indices:

```cpp
tf::index_map_buffer<int> point_mapping;
point_mapping.f().allocate(original_size);
point_mapping.kept_ids().allocate(filtered_size);

// Use with reindexing operations
auto reindexed_data = tf::reindexed(original_data, point_mapping);

// Access underlying data
int* forward_map = point_mapping.f().data();
int* kept_indices = point_mapping.kept_ids().data();
```

## Policies

The `trueform` policy system allows you to compositionally add semantic information and behavior to primitives and ranges. This is achieved through two main operations, `tag` and `zip`, which are designed to feel like building up an object with a clean, expressive pipeline.

A policy operation maps an object to a new version with an enriched policy:
> object_t<Policy> -> object_t<new_policy<Policy>>

These operations are **hierarchy-idempotent**: applying the same policy twice has no additional effect, making them safe to use in generic code.

> **NOTE:** Topological policies will be presented in [Topology](#topology).

### Policy tag

A `tag` applies a single piece of metadata to an entire object, whether it's a single primitive or a range of primitives. We support tagging with an `id`, `normal`, `plane`, and `state`.

This enables you to build up a complex object on-the-fly. For example, here we create a single point and enrich it with an ID, a normal (which itself has an ID), and a composite state made from a direction and a color.

```cpp
auto base_pt = tf::random_point<float, 3>();
auto point_normal = tf::normalized(tf::random_vector<float, 3>());
auto direction = tf::normalized(tf::random_vector<float, 3>());
std::array<float, 3> color;
auto point = base_pt
                | tf::tag_id("point")
                | tf::tag_normal(point_normal | tf::tag_id("normal"))
                | tf::tag_state(direction, color);
```
Even after being enriched, the `point` is still a `point_like<3, Policy>` and can be used in any geometric calculation. Policies are preserved and correctly transformed through transformations (i.e. the normal and the direction component of the state are transformed).
```cpp
// still behaves like a point
point += tf::random_vector<float, 3>();
auto d2 = tf::distance2(point, tf::random_point<float, 3>());
// Transformations correctly handle the object and its policies.
tf::frame<float, 3> frame = tf::random_transformation<float, 3>();
auto transformed_point = tf::transformed(point, frame);

const auto &id = transformed_point.id();
const auto &transformed_normal = transformed_point.normal();
const auto &transformed_normal_id = transformed_normal.id();
const auto &[transformed_direction, same_color] = transformed_point.state();
```

### Policy zip

A `zip` operation applies per-element data to a `primitive_range`. It effectively "zips" a range of data (like normals or IDs) with the range of primitives, so that each primitive in the range gets its own corresponding piece of data, i.e.:
```cpp
primitive_range_t<..., zip_x<Policy>>::reference =
     tag_x<primitive_range<Policy>::reference>
```

We support zipping with `ids`, `normals`, and `states` on [primitive range](#primitive-ranges).
For example, to create`tf::points` where each point has its own `normal`, `id` and a `state` consisting of a direction and a color array:

```cpp
std::vector<float> raw_pts;
std::vector<float> raw_normals;
std::vector<std::string> ids;
std::vector<float> raw_direction;
std::vector<std::array<float, 3>> colors;

auto points =
    tf::make_points<3>(raw_pts)
    | tf::tag_id("enriched points")
    | tf::zip_ids(ids)
    | tf::zip_normals(tf::make_unit_vectors<3>(raw_normals))
    | tf::zip_states(tf::make_unit_vectors<3>(raw_direction), colors);

const auto & [directions_, colors_] = points.states();
const auto &[direction_, color_] = points.states().front();
const auto &id_ = points.id();
const auto &ids_ = points.ids();
const auto &normals_ = points.normals();
```

The `points` are still a `tf::points<Policy>` and behave like one
```cpp
auto transformed_point = tf::transformed(points.front(), frame);
const auto &id = transformed_point.id();
const auto &transformed_normal = transformed_point.normal();
const auto &[transformed_direction, color_t] = transformed_point.state();
```

### Plain primitives and primitive ranges

To remove all policies from an object:

```cpp
auto plain_points = points | tf::plain();
```
## Algorithms

The core module provides parallel algorithms that integrate with `trueform`'s range and buffer systems.

### Basic Parallel Operations

#### Parallel Apply

Apply operations to ranges in parallel:

```cpp
// Basic parallel apply
tf::parallel_apply(points, [](auto&& pt) {
    pt = tf::normalized(pt.as_vector());
});

// With zip for multiple ranges
tf::parallel_apply(tf::zip(range1, range2), [](auto pair) {
    auto &&[elem1, elem2] = pair;
    elem1 += elem2;
});

// With checked execution for verification
tf::parallel_apply(points, [](auto&& pt) {
    pt = tf::normalized(pt.as_vector());
}, tf::checked);
```

#### Parallel Transform

Transform data in parallel:

```cpp
tf::buffer<float> results;
results.allocate(input_range.size());

tf::parallel_transform(input_range, results, [](auto value) {
    return value * value;  // Square each element
});
```

#### Parallel Copy Operations

```cpp
// Basic parallel copy
tf::parallel_copy(source_range, destination_range);

// Blocked copy for structured data
tf::parallel_copy_blocked(blocked_source, blocked_destination);

// Copy with index mapping
tf::parallel_copy_by_map_with_nones(source, destination, index_map, sentinel_value);
```

#### Parallel Fill and Utilities

```cpp
tf::buffer<int> data;
data.allocate(1000);

// Fill with value
tf::parallel_fill(data, 42);

// Generate sequence
tf::parallel_iota(data, 0);  // 0, 1, 2, 3, ...

// Replace values
tf::parallel_replace(data, old_value, new_value);
```

### Data Generation Algorithms

These algorithms create new data structures from input data, optimized for parallel execution and memory efficiency.

#### Generic Generate

`tf::generic_generate` provides parallel generation of variable-length data with automatic thread-local state management:

```cpp
// Generate boundary edges from faces using generic_generate
tf::blocked_buffer<int, 2> boundary_edges;
tf::generic_generate(tf::enumerate(faces), boundary_edges.data_buffer(),
    [&](const auto& pair, auto& buffer) {
        const auto& [face_id, face] = pair;
        int size = face.size();
        int prev = size - 1;

        for (int i = 0; i < size; prev = i++) {
            // Check if edge is on boundary
            if (is_boundary_edge(face[prev], face[i])) {
                buffer.push_back(face[prev]);
                buffer.push_back(face[i]);
            }
        }
    });

// Generate with thread-local state to avoid allocations
tf::buffer<int> result_data;
tf::generic_generate(input_range, result_data,
    tf::small_vector<int, 10>{},  // Thread-local work buffer
    [&](const auto& element, auto& output, auto& work_buffer) {
        work_buffer.clear();
        // Use work_buffer for intermediate calculations
        process_element(element, work_buffer);

        for (auto value : work_buffer) {
            output.push_back(value);
        }
    });

// Generate into multiple buffers simultaneously
auto buffers = std::tie(buffer_a, buffer_b, buffer_c);
tf::generic_generate(input_range, buffers,
    [&](const auto& element, auto& outputs) {
        auto& [out_a, out_b, out_c] = outputs;

        if (condition_a(element)) out_a.push_back(process_a(element));
        if (condition_b(element)) out_b.push_back(process_b(element));
        if (condition_c(element)) out_c.push_back(process_c(element));
    });
```

#### Generate Offset Blocks

`tf::generate_offset_blocks` creates offset-block data structures efficiently from input ranges:

```cpp
// Build face adjacency using generate_offset_blocks
tf::buffer<int> adjacency_offsets;
tf::buffer<int> adjacency_data;

tf::generate_offset_blocks(tf::make_sequence_range(faces.size()),
                          adjacency_offsets, adjacency_data,
    [&](int face_id, auto& buffer) {
        // Generate variable-length adjacency list for each face
        for (auto neighbor_id : compute_face_neighbors(face_id)) {
            buffer.push_back(neighbor_id);
        }
    });

// or work directly with the offset block buffer
tf::offset_block_buffer<int, int> result;
tf::generate_offset_blocks(input_range, result, work_lamda);
```

#### Blocked Reduce

`tf::blocked_reduce` provides efficient parallel reduction with custom aggregation logic:

```cpp
// Build intersection points using blocked_reduce
tf::buffer<tf::intersect::simple_intersection<int>> intersections;
tf::buffer<tf::point<float, 3>> intersection_points;

tf::blocked_reduce(
    tf::enumerate(polygons),
    std::tie(intersections, intersection_points),
    std::make_tuple(tf::buffer<tf::intersect::simple_intersection<int>>{},
                    tf::buffer<tf::point<float, 3>>{}),
    [&](const auto& polygon_range, auto& local_result) {
        auto& [local_intersections, local_points] = local_result;

        // Process polygon range in parallel
        for (const auto& [polygon_id, polygon] : polygon_range) {
            compute_intersections(polygon, local_intersections, local_points);
        }
    },
    [&](const auto& local_result, auto& global_result) {
        auto& [local_intersections, local_points] = local_result;
        auto& [global_intersections, global_points] = global_result;

        // Aggregate local results into global buffers
        auto old_size = global_intersections.size();
        global_intersections.reallocate(old_size + local_intersections.size());
        std::copy(local_intersections.begin(), local_intersections.end(),
                  global_intersections.begin() + old_size);

        old_size = global_points.size();
        global_points.reallocate(old_size + local_points.size());
        std::copy(local_points.begin(), local_points.end(),
                  global_points.begin() + old_size);
    });
```

#### Blocked Reduce with Sequenced Aggregation

`tf::blocked_reduce_sequenced_aggregate` ensures aggregation happens in sequential order, critical for operations where order matters. Bellow is the same implementation of adjecancy we used `tf::generate_offset_blocks` for (which is a wrapper around `tf::blocked_reduce_sequenced_aggregate`).

```cpp
// Build face adjacency per edge with ordered aggregation
tf::buffer<int> offsets;
tf::buffer<int> adjacency_data;

tf::blocked_reduce_sequenced_aggregate(
    tf::enumerate(faces),
    std::tie(offsets, adjacency_data),
    std::make_pair(tf::buffer<int>{}, tf::buffer<int>{}),
    [&](const auto& face_range, auto& local_result) {
        auto& [local_offsets, local_data] = local_result;

        // Process faces in parallel
        for (const auto& [face_id, face] : face_range) {
            for (int i = 0; i < face.size(); ++i) {
                local_offsets.push_back(local_data.size());
                // Add edge neighbors to local_data
                add_edge_neighbors(face_id, face[i], face[(i+1) % face.size()],
                                  local_data);
            }
        }
    },
    [&](const auto& local_result, auto& global_result) {
        auto& [local_offsets, local_data] = local_result;
        auto& [global_offsets, global_data] = global_result;

        // Sequential aggregation preserves order
        auto old_data_size = global_data.size();
        global_data.reallocate(old_data_size + local_data.size());
        std::copy(local_data.begin(), local_data.end(),
                  global_data.begin() + old_data_size);

        auto old_offsets_size = global_offsets.size();
        global_offsets.reallocate(old_offsets_size + local_offsets.size());
        auto it = global_offsets.begin() + old_offsets_size;
        for (auto local_offset : local_offsets) {
            *it++ = local_offset + old_data_size;
        }
    });

// Sequential aggregation ensures offset consistency
offsets.push_back(adjacency_data.size());
```
