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

Additionally, they support comparison operators.

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

// Double injection is a no-op
auto polygon5 = tf::inject_ids(ids, polygon4);
static_assert(std::is_same_v<decltype(polygon4), decltype(polygon5)>);

// They all view the same data
assert(&r[0][0] == &polygon4[0][0]);
```

Policy injections on primitives are preserved under transformations, allowing behavior to remain consistent as geometry moves through space.

## Frames and Transformations
