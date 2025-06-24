# Tutorial

## Core

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

### Primitives

> **primitive** → primitive_range → form

`trueform` provides a set of geometric primitives, including points, vectors, lines, polygons, AABBs, and transformations. Every primitive is a template of the form:
```c++
primitive<std::size_t(Dims), Policy>
```
The `Policy` parameter is the engine of the library, defining the primitive's behavior (e.g., whether it owns its data or just views it, does it have a normal, does it have state, etc) and enabling extensions. The underlying coordinate type (e.g., float, double) can be extracted from any policy:

```c++
// Deduces the scalar type from one or more policies
using common_t = tf::coordinate_type_t<Policy0, Policy1>;
common_t scalar = 2.0;
```

> **See:** All primitives may be tagged with additional semantics, as we will learn when we learn about [policies](#policies).

#### Points and Vectors

These are the most fundamental primitives, provided in three main variations:

| Concept         | General Template                     | Owning Alias                  | View Alias                         |
| --------------- | ------------------------------------ | ----------------------------- | ---------------------------------- |
| **Vector**      | `tf::vector_like<Dims, Policy>`      | `tf::vector<Type, Dims>`      | `tf::vector_view<Type, Dims>`      |
| **Unit Vector** | `tf::unit_vector_like<Dims, Policy>` | `tf::unit_vector<Type, Dims>` | `tf::unit_vector_view<Type, Dims>` |
| **Point**       | `tf::point_like<Dims, Policy>`       | `tf::point<Type, Dims>`       | `tf::point_view<Type, Dims>`       |

Factory functions like `tf::make_vector` and `tf::make_vector_view` create these primitives.

```c++
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

#### Line and Ray

Lines and rays are a composite of a point `origin` and a vector `direction`.

| Concept         | General Template                     | Owning Alias                  | 
| --------------- | ------------------------------------ | ----------------------------- | 
| **Line**        | `tf::line_like<Dims, Policy>`        | `tf::line<Type, Dims>`        |
| **Ray**         | `tf::ray_like<Dims, Policy>`         | `tf::ray<Type, Dims>`         |


There are no view aliases because a call to `make_line_like(point_like, vector_like)` will produce a view, when
the inputs are views. The factory `make_line` always returns an owning alias, as does `make_line_between_points`. Furthermore, the `line_like` is convertible to a line. The same holds true for rays.

They additionally support an `operator()(tf::coordinate_type<Policy> t): origin + t * direction`

#### Plane

A plane is a composite of a `unit_vector_like<Dims, Policy>` normal and a `tf::coordinate_type<Policy>` d, where `d` is the negative dot product between the normal and a point in the plane.

| Concept          | General Template                      | Owning Alias                   | 
| ---------------- | ------------------------------------- | ------------------------------ | 
| **plane**        | `tf::plane_like<Dims, Policy>`        | `tf::plane<Type, Dims>`        |


There are no view aliases because a call to `make_plane_like(unit_vector_like, coordinate_type)` will produce a view, when
the inputs are views. The factory `make_plane` always returns an owning alias.

```c++
tf::point<float, 3> pt;
tf::unit_vector<float, 3> normal;
auto plane0 = tf::make_plane(pt, pt, pt);
auto plane1 = tf::make_plane(normal, pt);
auto plane2 = tf::make_plane(normal, -tf::dot(normal, pt));
```

Furthermore, the `plane_like` is convertible to a `tf::plane`

#### Segment

A `tf::segment<Dims, Policy>` is a wrapper around a policy that behaves as a range of `tf::point_like`, with compile-time size of `2`. There are several ways to create one:

```c++
// creates copies of the two points
auto seg0 = tf::make_segment_between_points(pt0, pt1);
auto [seg0_pt0, seg0_pt1] = seg0;

std::array<tf::point<float, 3>, 2> r; // any range of two points
auto seg1 = tf::make_segment(r);
```

Or, if you have edges that index into a larger range of points:

```c++
std::array<int, 2> ids{0, 1};
auto seg2 = tf::make_segment(ids, r);

// ids are a view, as are the points
auto [id0, id1] = seg2.indices();
auto [seg2_pt0, seg2_pt1] = seg2;
```

#### Polygon

A `tf::polygon<Dims, Policy>` is a wrapper around a policy that behaves as a range of `tf::point_like`. When the policy has has a static size, so does the polygon. There are several ways to create one:

```c++
std::array<tf::point<float, 3>, 3> r;
auto polygon0 = tf::make_polygon(r);
```

Or, if you have faces that index into a larger range of points:

```c++
std::array<int, 3> ids{0, 1, 2};
auto polygon1 = tf::make_polygon(ids, r);

// ids are a view, as are the points
auto [id0, id1, id2] = polygon1.indices();
auto [pt0, pt1, pt2] = seg2;
```

> **NOTE**: Polygons use `tf::static_size` internally to determine the static size of the range being passed into it. All `tf` ranges propagate this static information. We still offer an overload `tf::make_polygon<V>` where the user manually supplies this information.

> **See:** [Policies](#policies) to see how to inject normals and planes into polygons.

#### AABB

An axis-aligned-bounding-box is a composite of a point_like `min` and a point_like `max`, representing the minimal and maximal corners.

| Concept         | General Template                     | Owning Alias                  | 
| --------------- | ------------------------------------ | ----------------------------- | 
| **aabb**        | `tf::aabb_like<Dims, Policy>`        | `tf::aabb<Type, Dims>`        |


There are no view aliases because a call to `make_aabb_like(point_like, point_like)` will produce a view, when
the inputs are views. The factory `make_aabb` always returns an owning alias. Furthermore, the `aabb_like` is convertible to a `tf::aabb`

It additionally supports `::diagonal()`, `::center()`, and `::size()` methods.

##### AABB of Primitives

To create an `aabb` of a finitie primitive:

```c++
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
```c++
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

There are no view aliases because a call to `make_frame_like(transformation_like, transformation_like)` will produce a view, when
the inputs are views. The factory `make_frame` always returns an owning alias.

```c++
tf::frame<float, 3> frame0;
tf::frame<float, 3> frame1{tr_view};
```
The `coordinate type` is again accessible through `tf::coordinate_type<Policy>`.

Additionally, `tf::frame` uses a safe policy that ensures the inverse is always computed after changes. This means that to alter the transformation in a frame, you must use 

```c++
frame.fill(float_ptr/*, ptr for inverse*/);
frame.set(transformation /*, inverse_transformation*/);
```

The general `tf::frame_like` offers no such guarantees, but is convertible to a `tf::frame`.

#### Transforming primitives

Any primitive from the `tf::` namespace can be transformed using `tf::transformed(_this, _by_transformation)`. Transformations can be performed using either `tf::transformation` or `tf::frame`. This includes composing transformations:

```c++
auto tr0 = tf::random_transformation<float>();
auto tr1 = tf::random_transformation<float>();
auto tr1_dot_tr0 = tf::transformed(tr0, tr1);
```

### Queries of Primitives

`trueform` provides a set of queries that work across all compatible primitive types.

#### General queries

All pairs of primitives support the following queries:

| **Query**                 | **Returns**                         |
|---------------------------|-------------------------------------|
|`distance2`                |Squared distance between primitives  |
|`distance`                 |Distance between primitives          |
|`closest_metric_point`     |`tf::metric_point` (on left argument)|
|`closest_metric_point_pair`|`tf::metric_point_pair`              |
|`intersects`               |Do the primitives intersect          |

A `tf::metric_point` is a composite of a `metric` and a `point`, while `tf::metric_point_pair` is a composite of a `metric`,  `point` first and `point` second.

```c++
auto [dist2, pt_on_poly0, pt_on_poly1] =
    tf::closest_metric_point_pair(polygon0, polygon1);
```

#### Point classification queries

A point may be classified (using `tf::classify`) against a plane and polygon, and additionally against a line, ray and segment in 2D. The function returns either a `tf::sidedness`, or a `tf::containment` enum, depending on the context.

#### Ray casting

We distinguish two types of ray operations: `ray_cast` (`intersection_status` and `t`) and `ray_hit` (`intersection_status`, `t` and the `tf::point` of the hit).

Both are supported between a ray and all other primitive types.

```c++
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

### Primitive Ranges

> primitive → **primitive_range** → form


`trueform` provides a set of semantic ranges over primitives. Every primitive_range is a template of the form:
```c++
primitive_range<std::size_t(Dims), Policy>
```
The `Policy` parameter defines the primitive_range's behavior (e.g. does it have normals, does it have state, does it have connectivity structures like vertex_link, etc) and enables extensions. The underlying coordinate type (e.g., float, double) can be extracted from any policy:

```c++
// Deduces the scalar type from one or more policies
using common_t = tf::coordinate_type_t<Policy0, Policy1>;
common_t scalar = 2.0;
```

#### General Range View Adaptors

While `trueform` is not a range-adaptor library, it provides several range adaptors that allow you to work with your existing data layout directly. Our adaptors preserve and propagate static size information, which can be accessed via `tf::static_size`. Here are a few common examples:

```c++
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

To create a view of your data:
```c++
auto r0 = tf::make_range(your_container);
// if you know it has n-elements and this
// is not known to tf::static_size
auto r1 = tf::make_range<N>(your_container);
```

#### Points and Vectors

These are the mose fundamental primitive ranges, provided in three variants:

|Concept     |Template                  |Factory          |
|------------|--------------------------|-----------------|
|vectors     |vectors<Dims, Policy>     |make_vectors     |
|unit vectors|unit_vectors<Dims, Policy>|make_unit_vectors|
|points      |points<Dims, Policy>      |make_points      |

The factories can take flat ranges as inputs:
```c++
std::vector<float> raw_pts;
auto pts = tf::make_points<3>(raw_pts);
tf::point_view<float, 3> pt = pts.front();
```
Or simply wrap ranges already containing the concerning primitive:
```c++
std::vector<tf::point<float, 3>> pts0;
auto pts = tf::make_points(pts0);
```

> **NOTE:** The `points<Dims, Policy>` provide an additional method `.as_vector_view()`, when one needs complete vector algebra over their points.

> **See:** Points and vectors may be tagged with additional sementics (normals, ids, state, topological connectivity, etc), as we will learn when we learn about [policies](#policies).

#### Segments

Curves and embedded graphs are modeled by a range of segments:

|Concept      |Template              |Factory          |
|-------------|----------------------|-----------------|
|segments     |segments<Dims, Policy>|make_segments    |

The factory is used in conjunction [General View Adaptors](#general-range-view-adaptors):

```c++
// from a sequence of edges
auto segments0 = tf::make_segments(tf::make_blocked_range<2>(points));
// from a sequence of points
auto segments1 = tf::make_segments(tf::make_slide_range<2>(points));

```
We can use ids to index into a larger array of points:
```c++
std::vector<int> ids;
auto segments2 = tf::make_segments(tf::make_blocked_range<2>(ids), points);
auto segments3 = tf::make_segments(tf::make_slide_range<2>(ids), points);
```
The segment elements behave [as expected](#segment):
```c++
auto [s3pt0, s3pt1] = segments3.front();
auto [s3_id0, s3_id1] = segments3.front().indices();
```
We additionally get two methods, `.edges()` and `.points()`:
```c++
auto [s3_id0_, s3_id1_] = segments3.edges().front();
auto points_range = segments3.points();
```

> **See:** Segments may be tagged with additional sementics (normals, ids, state, topological connectivity, etc), as we will learn when we learn about [policies](#policies).

#### Polygons

Meshes are modeled by a range of segments:

|Concept      |Template              |Factory          |
|-------------|----------------------|-----------------|
|polygons     |polygons<Dims, Policy>|make_polygons    |

The factory is used in conjunction [General View Adaptors](#general-range-view-adaptors):
```c++
// from a sequence of points
auto polygons0 = tf::make_polygons(tf::make_blocked_range<3>(points));
auto [p0pt0, p0pt1, p0pt2] = polygons0.front();
```
We can use ids to index into a larger array of points:
```c++
std::vector<int> ids;
auto polygons1 = tf::make_polygons(tf::make_blocked_range<3>(ids), points);
// or if the ids are in legacy VTK format (3, a, b, c, 3, e, f, g)
auto polygons2 =
    tf::make_polygons(tf::make_tag_blocked_range<3>(ids), points);
```
The polygons elements behave [as expected](#polygon):
```c++
auto [p2pt0, p2pt1, p2pt2] = polygons2.front();
auto [p2_id0, p2_id1, p2_id2] = polygons2.front().indices();
```
We additionally get two methods, `.faces()` and `.points()`:
```c++
auto [p2_id0_, p2_id1_, p2_id2_] = polygons2.faces().front();
auto points_range = polygons2.points();
```

> **See:** Polygons may be tagged with additional sementics (normals, ids, state, topological connectivity, etc), as we will learn when we learn about [policies](#policies).

### Policies

To achieve *lamdba-like*, inline class-building, we support policy tagging and zipping on [primitives](#primitives) and [primitive_ranges](#primitive-ranges).

A policy injection like `inject_x` composes additional behavior onto an object by wrapping its existing policy.  
It maps a type `object_t<..., Policy>` to a new version with added semantics:

```c++
inject_x: object_t<..., Policy> -> object_t<..., inject_x<Policy>>
```

Policy injections are **idempotent**:

```c++
(inject_x)^2 = inject_x
```

Injecting the same behavior twice is a no-op.


## Spatial

## Topology




### Fixed Primitives

These primitives have a fixed internal structure and do not support policy injection. They are:

* `tf::ray<Type, Dims>`
* `tf::line<Type, Dims>`
* `tf::plane<Type, Dims>`
* `tf::aabb<Type, Dims>`

They are primarily used for queries such as ray casting, line intersection, or point projection. Each comes with its own factory:

```c++
auto ray0 = tf::make_ray(origin, direction);
auto ray1 = tf::make_ray_between_points(pt0, pt1);
auto line0 = tf::make_ray(origin, direction);
auto line1 = tf::make_line_between_points(p0, p1);
auto plane0 = tf::make_plane(pt0, pt1, pt2);
auto plane1 = tf::make_plane(unit_vector, pt);
auto aabb0 = tf::make_aabb(min_pt max_pt);
auto aabb1 = tf::aabb_from(finite_primitive);
```

These types support relevant operations but are not part of the policy/composable hierarchy.

## Semantic Views

`primitive -> **semantic_view** -> form`

Primitives are joined into ranges (e.g., `tf::points`, `tf::segments`, `tf::polygons`) that have additional semantics injected via policies.

One rarely operates on individual primitives, but on a range of them, i.e., `tf::points`, `tf::segments`, `tf::polygons` etc.

### Range Adaptors

While `trueform` is not a range library, it provides several range adaptors that allow you to work with your existing data layout directly. These adaptors enable efficient traversal, transformation, and indexing of data in a composable way.


- **`blocked_range`**  
  - Call: `make_blocked_range` (static or dynamic block size)
  - Example: `make_blocked_range<2>([0, 1, 2, 3])` → `[[0, 1], [2, 3]]`
  - Use Case: Interpreting polygon or segment vertex IDs

- **`tag_blocked_range`**  
  - Call: `make_tag_blocked_range` (static or dynamic block size)
  - Example: `make_tag_blocked_range<2>([t0, 0, 1, t1, 2, 3])` → `[[0, 1], [2, 3]]`
  - Use Case: For layouts such as those used in legacy VTK (<8) polygon buffers

- **`slide_range`**  
  - Call: `make_slide_range` (static or dynamic window size)
  - Example: `make_slide_range<2>([0, 1, 2])` → `[[0, 1], [1, 2]]`
  - Use Case: Converting a curve to segment IDs (sliding window)

- **`indirect_range`**  
  - Call: `make_indirect_range(ids, data_range)`
  - Example: `make_indirect_range([1, 3], [a, b, c, d])` → `[b, d]`
  - Use Case: Offsetting into a point array by primitive vertex IDs

- **`block_indirect_range`**  
  - Call: `make_block_indirect_range(blocked_range, data)`
  - Example: `make_block_indirect_range([[0, 1], [2, 3]], [a, b, c, d])` → `[[a, b], [c, d]`
  - Use Case: Indirect into point using blocked polygon or segment vertex IDs

- **`mapped_range`**  
  - Call: `make_mapped_range(range, f)`
  - Example: `make_mapped_range([0, 1, 2], f)` → `[f(0), f(1), f(2)]`
  - Use Case: Applying functions to every element in a range

- **`sequence_range`**  
  - Call: `make_sequence_range(start, end)`
  - Example: `make_sequence_range(3, 6)` → `[3, 4, 5]`
  - Use Case: Efficient for loop-style iteration or index generation

- **`offset_blocked_range`**  
  - Call: `make_offset_blocked_range(offsets, data)`
  - Example: `make_offset_blocked_range([0, 2, 3, 5], [a, b, c, d, e])` → `[[a, b], [c], [d, e]]`
  - Use Case: Offset layout for unstructured lists

> **Note**: `trueform` range adaptors propagate static size information where it is known at compile time.

```c++
auto [a, b, c] = tf::make_blocked_range<3>(r).front();
std::array<int, 2> ids{0, 1}; // as ids
for(auto [a, b]: tf::make_indirect_range(ids, r));
for(auto [id0, id1]: tf::make_indirect_range(ids, r).ids());
```

### Views of Primitives

Range adaptors are primarily used to prepare views for constructing geometric primitives into collections. They allow compositional definition of complex data without introducing custom wrappers or data duplication. Furthermore, all ranges of primitives support policy injection.

#### Points and Vectors

Points and vectors are created using factory functions.

```c++
std::vector<float> raw_data;
auto points = tf::make_points<Dims>(raw_data);
auto vectors = tf::make_vectors<Dims>(raw_data);
// does not normalize
auto unit_vectors = tf::make_unit_vectors<Dims>(raw_data)

tf::vector<float, Dims> sum = vectors.front() + vectors.back();
```

These are ranges over `point|vector|unit_vector` views over raw data. `tf::points` additionally provide a conversion method `as_vector_views()`, when one needs vector algebra over their points (like computing a centroid).
```c++
auto vector_views = points.as_vector_views();
```

#### Segments

Segments are created using the factory `tf::make_segments`. 

- **Without ids**: Assume we have a sequence of points representing a curve.
```c++
auto segments0 = tf::make_segments(tf::make_blocked_range<2>(points));
auto [pt0, pt1] = segments0.front();
```
- **With ids:** Assume ids `std::vector<int> ids;`
    - Assume the ids are a sequence of point ids representing a curve
    ```c++
    auto segments1 = tf::make_segments(tf::make_slide_range<2>(ids), points);
    auto [id0, id1] = segments1.front().ids();
    auto [id0_, id1_] = segments1.edges().front();
    auto points_view = segments1.points();
    ```
    - Assume the ids are a flat sequence of ids of segments
    ```c++
    auto segments2 = tf::make_segments(tf::make_blocked_range<2>(ids), points);
    auto [id0, id1] = segments2.front().ids();
    auto [id0_, id1_] = segments2.edges().front();
    auto points_view = segments2.points();

#### Polygons

Polygons are created using the factory `tf::make_polygons`. 

- **Without ids**: Assume we have a sequence of points `points`.
    - Assume points represent a sequence of triangles
    ```c++
    auto polygons0 = tf::make_polygons(tf::make_blocked_range<3>(points));
    auto [pt0, pt1, pt2] = polygons0.front();
    ```
    - Assume points represent a strip of triangles
    ```c++
    auto polygons1 = tf::make_polygons(tf::make_slide_range<3>(points));
    ```
- **With ids:** Assume ids `std::vector<int> ids;`
    - Assume the ids are a flat sequence of ids of triangles
    ```c++
    auto polygons2 = tf::make_polygons(tf::make_blocked_range<3>(ids), points);
    auto [id0, id1, id2] = polygons2.front().ids();
    auto [id0_, id1_, id2_] = polygons2.faces().front();
    auto points_view = polygons2.points();
    ```
    - Assume ids are tagged like in legacy VTK layout, i.e. `[3, a, b, c, 3, e, f, g, ...]`
    ```c++
    auto polygons3 = tf::make_polygons(tf::make_tag_blocked_range<3>(ids), points);
    ```
    - Assume the ids represent a strip of triangles ids
    ```c++
    auto polygons4 = tf::make_polygons(tf::make_slide_range<3>(ids), points);
    ```
- **With ids and normals:** Additionally assume `auto normals = tf::make_unit_vectors(r)`
    ```c++
    auto polygons5 = tf::make_polygons(tf::make_blocked_range<3>(ids), points, normals);
    auto [id0, id1, id2] = polygons5.front().ids();
    auto normal = polygons5.front().normal();
    auto [id0_, id1_, id2_] = polygons5.faces().front();
    auto points_view = polygons5.points();
    auto unit_vectors = polygons5.normals()
    ```

## Policy Injections

To achieve *lamdba-like*, inline class-building, we support policy injections into primitives and semantic views.

A policy injection like `inject_x` composes additional behavior onto an object by wrapping its existing policy.  
It maps a type `object_t<..., Policy>` to a new version with added semantics:

```c++
inject_x: object_t<..., Policy> -> object_t<..., inject_x<Policy>>
```

Policy injections are **idempotent**:

```c++
(inject_x)^2 = inject_x
```

Injecting the same behavior twice is a no-op.

### Mechanics

An `injectable class` implements two free functions:
- `unwrap: injectable_class<..., Policy> -> Policy`
- `wrap_like: (injectable_class<..., Policy0>, Policy1) -> injectable_class<..., Policy1>`


Each injection provides a `has_injected_X<Type>` compile time check, if the policy is already present in the `Type`.

These are used by an injection like so:
```c++
if constexpr(has_injected_X<decltype(injectable)>)
    return injectable;
auto && policy = unwrap(injectable);
return wrap_like(injectable, inject_X(policy));
```
> **NOTE:** `wrap` and `unwrap_like` default to the identity function.

These policy injections are a core part of `trueform`’s design philosophy — enabling inline, composable expressions without boilerplate. We will introduce injections on primitives here and injections on ranges of primitives when we get to them.

### Policy injections on primitives

The following policy injections are provided for primitives:

- `inject_id(id, object)`  
  Adds an `id()` method to the object, returning the given scalar ID.

- `inject_ids(id_range, object)`  
  Adds an `ids()` method, returning a view of the ID range (e.g. vertex indices of a polygon).

- `inject_normal(unit_vector, object)`  
  Adds a `normal()` method returning the supplied unit vector. If the object is a polygon, the normal can be computed automatically.

- `inject_plane(plane, object)`  
  Adds both `plane()` and `normal()` methods. For polygons, the plane can also be computed if not provided.

### Example on a polygon

```c++
std::array<tf::point<float, 3>, 3> r; // any range of 3 points
using policy0_t = decltype(r);
tf::polygon<3, policy0_t> polygon0 = tf::make_polygon(r);

std::array<int, 3> ids{2, 3, 4};
using ids_t = decltype(ids);
using policy1_t = tf::inject_ids_t<ids_t, policy0_t>;

// Injects a copy of ids
tf::polygon<3, policy1_t> polygon1 = tf::inject_ids(ids, polygon0);

// Injects a view of ids
auto polygon1_id_view = tf::inject_ids(tf::make_range(ids), polygon0);

// Because make_range propagates static size information
auto [id0, id1, id2] = polygon1_id_view.ids();

tf::unit_vector<float, 3> normal0{{1, 0, 0}};
auto polygon_with_normal = tf::inject_normal(normal0, polygon1);

// Or have it be computed
auto polygon2 = tf::inject_normal(polygon1);

// You can also inject a plane
auto polygon3 = tf::inject_plane(polygon1);

// This detects the already injected normal
auto polygon4 = tf::inject_plane(polygon2);
const tf::plane<float, 3> &plane0 = polygon3.plane();

// They all view the same data
assert(&r[0][0] == &polygon4[0][0]);
```

> **Note**: Because injections are idempotent, you can safely inject a plane in any generic context — whether or not it has already been computed:
```c++
auto polygon5 = tf::inject_plane(polygon4);
static_assert(std::is_same_v<decltype(polygon4), decltype(polygon5)>);
```
> The compiler will resolve whether the injection is necessary. No redundant computations are performed.

Policy injections on primitives are preserved under transformations, allowing behavior to remain consistent as geometry moves through space.


## Policy Injections: Primitive Ranges

Policy injections on primitive ranges operate in the same manner as [those on primitives](#policy-injections-primitives). We will return to them after we learn about the spatial and connectivity structures.





## Frames and Transformations

A `tf::transformation<RealType, Dims>` consists of a rotation `R` and a translation `t`. It maps points and vectors according to the table below:

| **Type**       | **Mapping**        |
|----------------|--------------------|
| `point_like`   | `R.dot(pt) + T`    |
| `vector_like`  | `R.dot(vec)`       |

A `tf::frame<RealType, Dims>` wraps a transformation and its inverse, effectively framing the object in the scene. The frame tracks changes to the transformation and computes the inverse on request.

```c++
tf::transformation<float, 3> transform = tf::random_transformation<float>();
// copies the transformation
tf::frame<float, 3> frame{transform};
const tf::transformation<float, 3> & inv_transform = frame.inverse_transformation();
```

Since `trueform` is often integrated into systems that already manage transformations, you can simply `fill` them in where needed:

```c++
// the last row of a Dims x Dims matrix is
// implicitly [0, ...., 1] 
float raw_transform[Dims * (Dims + 1)];
transform.fill(raw_transform);
// this sets the inverse as dirty
frame.fill(raw_transform);
// as does this
frame = transform;
```

If you already have the inverse computed, you may set it together with the transformation:

```c++
frame.fill(raw_transform, raw_inverse_transform);
frame.set(transform, inverse_transform);
```

We also provide two convenience factories: `tf::make_identity_transformation` and `tf::make_transformation_from_translation`.

### Transforming Primitives

Any primitive from the `tf::` namespace can be transformed using `tf::transformed(_this, _by_transformation)`. Transformations can be performed using either `tf::transformation` or `tf::frame`. This includes composing transformations:

```c++
auto tr0 = tf::random_transformation<float>();
auto tr1 = tf::random_transformation<float>();
auto tr1_dot_tr0 = tf::transformed(tr0, tr1);
```

### Transformations and Injectable Policies

Transformations via `tf::frame` always preserve injectable policies.  
Transformations via `tf::transformation` may drop `inject_plane_t` or `inject_normal_t` when inverses are needed but not available (e.g. for points and segments). For this reason, transforming with a frame is generally more efficient.

#### Example

```c++
std::array<tf::point<float, 3>, 6> pts{};
using base_t = decltype(pts);
std::array<int, 3> ids{3, 4, 5};
auto poly0 = tf::make_polygon(ids, pts);
auto poly1 = tf::inject_plane(poly0);
auto poly2 = tf::inject_id(0, poly1);

using point_holder_t = std::array<tf::point<float, 3>, 3>;
tf::polygon<
    3, tf::inject_id_t<
            int, tf::inject_plane_t<float, 3,
                                    tf::inject_ids_t< //
                                        tf::range<int *, 3>, point_holder_t>>>>
    transformed_poly = tf::transformed(poly2, fr);
```

> **Note**: `poly2` indexed into `pts` using IDs. After transformation, it holds the three transformed points and still views the original ID range.

## Spatial Structures

We distinguish between two types of spatial structures: those that support modification (e.g., for live editing of a mesh) and those that do not.

### `tf::tree`

`tf::tree<Index, RealType, Dims>` is a high-performance spatial hierarchy designed for real-time geometry queries, introduced in:

> Žiga Sajovic, Dejan Knez, and Robert Korez.  
> **tf::tree: A General-Purpose Spatial Hierarchy for Real-Time Geometry Queries.**  
> *Institute of Electrical and Electronics Engineers (IEEE),* June 2025.  
> [https://doi.org/10.36227/techrxiv.174952959.92977743/v1](https://doi.org/10.36227/techrxiv.174952959.92977743/v1)

#### Building the Tree

To construct a tree, provide a [primitive range](#primitive-ranges) and a configuration specifying the arity (number of children per node) and the maximum number of primitives in a leaf.

```c++
tf::tree<int, float, 3> tree(primitive_range, tf::config_tree(4, 4));
// or
tree.build(primitive_range, tf::config_tree(4, 4));
```

Empirically, the configuration `(4, 4)` offers the best performance across workloads.

To support custom primitive types, pass a lambda that returns an `tf::aabb`. This is the default:

```c++
tf::config_tree(4, 4, [](const auto& primitive) {
    using tf::aabb_from;
    return aabb_from(primitive);
});
```

#### Partitioning Policies

Tree construction uses a partitioning strategy analogous to `std::nth_element`. We support several [selection-based algorithms](https://github.com/danlark1/miniselect):

- `tf::strategy::floyd_rivest`  
- `tf::strategy::pdq`  
- `tf::strategy::median_of_medians`  
- `tf::strategy::median_of_ninthers`  
- `tf::strategy::median_of_3_random`  
- `tf::strategy::heap_select`  
- `tf::strategy::nth_element` *(default)*

Example:

```c++
tree.build(tf::strategy::nth_element, primitive_range, tf::config_tree(4, 4));
```

#### Queries on the Tree

Tree queries are described in the [Spatial Queries](#spatial-queries) section, after introducing [Forms](#form).

---

### `tf::mod_tree`

`tf::mod_tree<Index, RealType, Dims>` extends `tf::tree` with an `update()` method, making it suitable for scenarios where the spatial layout changes dynamically, like free-forming.

## Forms

`primitive -> semantic_view -> **form**`

## Spatial Queries

## Topology and Connectivity Structures

## Algorithms

## Utilities
