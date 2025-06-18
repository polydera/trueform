# Tutorial

## Primitives

`trueform` supports various geometric primitives. These can be grouped into two categories:

* **Injectable primitives**: support composable policies like IDs, planes, normals.
* **Fixed primitives**: do not support policy injection; they have fixed structure and behavior.

### Injectable Primitives

These include points, vectors, segments, and polygons. They allow semantic enrichment through policy injection while maintaining lightweight views.

#### Points and Vectors

Points and vectors are the basic primitives of geometry. We implement them through three main classes:

| Concept         | General Template                     | Owning Alias                  | View Alias                         |
| --------------- | ------------------------------------ | ----------------------------- | ---------------------------------- |
| **Vector**      | `tf::vector_like<Dims, Policy>`      | `tf::vector<Type, Dims>`      | `tf::vector_view<Type, Dims>`      |
| **Unit Vector** | `tf::unit_vector_like<Dims, Policy>` | `tf::unit_vector<Type, Dims>` | `tf::unit_vector_view<Type, Dims>` |
| **Point**       | `tf::point_like<Dims, Policy>`       | `tf::point<Type, Dims>`       | `tf::point_view<Type, Dims>`       |

Owning variants use the policy `tf::owned_data<Type, Dims>`, and views use `tf::borrowed_data<Type, Dims>`.

##### Creation

Each of them comes with a factory function `tf::make_X`, where X is any of the above.

```c++
tf::vector<float, 3> v0{{1, 1, 1}};
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

##### Conversions

All concepts support `::as<T>()` method that returns a copy with values of type `T`, and a conversion operator.

```c++
tf::vector<double, 3> dv0 = v0.as<double>();
tf::vector<double, 3> dv1 = v0;
```

Points can be converted to `vector_view<T, Dims>`.

```c++
tf::vector_view<float, 3> vview2 = pt0.as_vector_view();
```

##### Algebra

Vectors support vector algebra:

```c++
v0 += 2 * view0 - vview0 + vview1;
```

Unit vectors prohibit assigning algebraic operations and return `tf::vector` for non-assigning operations:

```c++
uv1 += uv2; // won't compile
tf::vector<float, 3> v1 = uv1 + uv2;
```

Points support the `operator-` between each other and `operator-` and `operator+` with vectors.

```c++
tf::vector<float, 3> v2 = pview0 - pview1;
tf::point<float, 3> v3 = pview0 + v2;
tf::point<float, 3> v4 = pview0 + (pview0 - pview1);
```

If you need algebra on points, such as computing a centroid, use the `::as_vector_view()` method.

Additionally, they support comparison operators and `tf::dot` and `tf::cross`.

#### Segment

A `tf::segment<Policy>` is a wrapper around a policy that defines an `operator[] -> tf::point_like`. There are several ways to create one:

```c++
auto seg0 = tf::make_segment_between_points(pt0, pt1);
auto [seg0_pt0, seg0_pt1] = seg0;

std::array<tf::point<float, 3>, 2> r; // any range of two points
auto seg1 = tf::make_segment(r);
```

Or, if you have edges that index into a larger range of points:

```c++
std::array<int, 2> ids{0, 1};
auto seg2 = tf::make_segment(ids, r);

// ids are a view, like the points are
auto [id0, id1] = seg2.ids();
auto [seg2_pt0, seg2_pt1] = seg2;
```

#### Polygon

A `tf::polygon<N_Vertices, Policy>` is a wrapper around a policy that defines an `operator[] -> tf::point_like`. There are several ways to create one:

```c++
std::array<tf::point<float, 3>, 3> r;
auto polygon0 = tf::make_polygon(r);
```

Or, if you have faces that index into a larger range of points:

```c++
std::array<int, 3> ids{0, 1, 2};
auto polygon1 = tf::make_polygon(ids, r);

// ids are a view, like the points are
auto [id0, id1, id2] = polygon1.ids();
auto [pt0, pt1, pt2] = seg2;
```

> **NOTE**: Polygons use `tf::static_size` internally to determine the static size of the range being passed into it. All `tf` ranges propagate this static information. We still offer an overload `tf::make_polygon<V>` where the user manually supplies this information.

**See**: [Policy injection on Primitives](#policy-injection-on-primitives) for a demonstration of handling normals, planes and ids.

### Fixed Primitives

These primitives have a fixed internal structure and do not support policy injection. They are:

* `tf::ray<Type, Dims>`
* `tf::line<Type, Dims>`
* `tf::plane<Type, Dims>`

They are primarily used for queries such as ray casting, line intersection, or point projection. Each comes with its own factory:

```c++
auto ray0 = tf::make_ray(origin, direction);
auto ray1 = tf::make_ray_between_points(pt0, pt1);
auto line0 = tf::make_ray(origin, direction);
auto line1 = tf::make_line_between_points(p0, p1);
auto plane0 = tf::make_plane(pt0, pt1, pt2);
auto plane1 = tf::make_plane(unit_vector, pt);
```

These types support relevant operations but are not part of the policy/composable hierarchy.


## Policy Injections

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

These policy injections are a core part of trueform’s design philosophy — enabling inline, composable expressions without boilerplate. We will introduce injections on primitives here and injections on ranges of primitives when we get to them.

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

## Primitive Ranges

One rarely operates on individual primitives, but on a collection of them, i.e., `tf::points`, `tf::segments`, `tf::polygons` etc.

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
```

### Ranges of Primitives

Range adaptors are primarily used to prepare views for constructing geometric primitives into collections. They allow compositional definition of complex data without introducing custom wrappers or data duplication.

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
    auto [pt0, pt1, pt2] = polygons1.front();
    ```
- **With ids:** Assume ids `std::vector<int> ids;`
    - Assume the ids are a flat sequence of ids of triangles
    ```c++
    auto polygons2 = tf::make_polygons(tf::make_blocked_range<3>(ids), points);
    auto [id0, id1, id2] = polygons2.front().ids();
    auto [id0_, id1_, id2_] = polygons2.faces().front();
    auto points_view = polygons2.points();
    ```
    - Assume ids are tagged likle in legacy VTK layout, i.e. `[3, a, b, c, 3, e, f, g, ...]`
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
