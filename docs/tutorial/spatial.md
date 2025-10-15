# Spatial

Spatial module extends the [Queries on Primitives](#queries-of-primitives) to primitive ranges by introducing [Forms](#forms), that bundle a [primitive range](#primitive-ranges), a [Frame](#transformations-and-frames) and a [Spatial structure](#spatial-structures).

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

- `tf::spatial::floyd_rivest_t`  
- `tf::spatial::pdq_t`  
- `tf::spatial::median_of_medians_t`  
- `tf::spatial::median_of_ninthers_t`  
- `tf::spatial::median_of_3_random_t`  
- `tf::spatial::heap_select_t`  
- `tf::spatial::nth_element_t` *(default)*

Example:

```c++
tree.build<tf::spatial::nth_element>(primitive_range, tf::config_tree(4, 4));
```

### `tf::mod_tree`

`tf::mod_tree<Index, RealType, Dims>` extends `tf::tree` with an `update()` method, making it suitable for scenarios where the spatial layout changes dynamically, like free-forming.

## Forms

> primitive → primitive_range → **form**

A `tf::form` is a composite of a [primitive_range](#primitive-ranges), a [spatial structure](#spatial-structures) and a [frame](#transformations-and-frames) (the frame may be omittted). It is used for spatial queries.

```c++
std::vector<float> raw_pts;
auto pts = tf::make_points<3>(raw_pts)
           | tf::tag_id("point cloud");
tf::tree<int, float, 3> tree(pts, tf::config_tree(4, 4));
tf::frame<float, 3> frame = tf::random_transformation<float, 3>();
auto form = tf::make_form(frame, tree, pts);
```
A form behaves like the [primitive range](#primitive-ranges) it wraps:
```c++
auto pt0 = form[0];
```

## Spatial Queries

We extend the [queries on primitives](#queries-of-primitives) to forms. All queries are supported between a `form` and a `primitive` and between two `forms`. The queries `tf::distance`, `tf::distance2` and `tf::intersects` are trivially extended:
```c++
auto d2_0 = tf::distance2(form0, form1);
auto d2_1 = tf::distance2(form0, form1[0]);
bool do_intersects_0 = tf::intersects(form0, form1);
bool do_intersects_1 = tf::intersects(form0, form1[0]);
```
The queries `closest_metric_point` and `closest_metric_point_pair` are extended via [Neighbor Search](#neighbor-search), and `ray_cast` and `ray_hit` via [Ray Casting on Forms](#ray-casting-on-forms). We additionally support a generic [Search](#search) query.

### Search
We support searching a form, searching over a pair of forms, and self-searching a form.

#### Searching a Form

A form is searched via:

```c++
tf::search(form, check_aabb_f, apply_to_primitive_f);
```
where `check_aabb_f: tf::aabb_like -> bool` is applied to aabbs in the tree, and `apply_to_primitive_f: primitive -> void | bool` is applied to individual primitives. If `apply_to_primitive_f` returns a boolean, the search stops on the first returned `true`. Additionally, the call to `tf::search` returns this boolean value.

For example, `tf::intersects(form, primitive)` is implemented like so:
```c++
tf::search(
    form,
    [obj_aabb = tf::aabb_from(primitive)](const auto &aabb) {
        return tf::intersects(aabb, point_like);
    },
    [&primitive](const auto &point) {
        return tf::intersects(point, primitive);
    });
```

While `gather_ids(form, predicate_f, out_iter)` is implemented like so:
```c++
tf::search(form, predicate, [&out_iter](const auto &obj) {
    if (predicate(obj))
        *outout_iter++ = obj.id();
    // note the absence of a return
});
```
> **NOTE:** if the primitive does not have an id policy, it is tagged with its index in the primitive range.

#### Searching a pair of Forms

A pair of forms is searched via:

```c++
tf::search(form0, form1, check_aabb_f, apply_to_primitives_f);
```
where `check_aabb_f: (tf::aabb_like, tf::aabb_like) -> bool` is applied to pairs of aabbs in the tree, and `apply_to_primitives_f: (primitive0, primitive1) -> void | bool` is applied to pairs of primitives. If `apply_to_primitives_f` returns a boolean, the search stops on the first returned `true`. Additionally, the call to `tf::search` returns this boolean value.

> **NOTE:** A search call over a pair of primitives is done in parallel. Hence, you must ensure thread-safety of the `apply_to_primitives_f`. Use `tf::local_value` and `tf::local_vector`.

For example, `tf::gather_ids(form0, form1, predicate, out_iter)` is implemented like so:
```c++
tf::local_vector<std::pair<id_t0, id_t1>> local;
tf::search(
    form, form, predicate,
    [&local, &predicate](const auto &primitive0, const auto &primitive1) {
        if (predicate(primitive0, primitive1))
            local.push_back(primitive0.id(), primitive1.id());
    });
local.to_iterator(out_iter);
```

#### Searching a Form against itself

A form is searched against itself via:

```c++
tf::search_self(form, check_aabb_f, apply_to_primitives_f);
```

where everything follows the behavior of [Searching a pair of Forms](#searching-a-pair-of-forms). This is useful for finding self intersections or epsilon-duplicates.

### Neighbor Search

`tf::neighbor_search` generalizes the `tf::closest_metric_point` and `tf::closest_metric_point_pair` queries between primitives. We support `tf::neighbor_search` between a form and a primitive and between a pair of forms.

```c++
auto nearest_neighbor =
    tf::neighbor_search(form, primitive /*, search_radius*/);
// when using a search_radius, nearest_neighbor might not exist
if (nearest_neighbor)
    auto [primitive_id, metric_point] = nearest_neighbor;

auto nearest_neighbor_pair =
    tf::neighbor_search(form0, form1, /*, search_radius*/);
if (nearest_neighbor_pair) {
    auto [primitive_ids, metric_point_pair] = nearest_neighbor_pair;
    auto [primitive_id0, primitive_id1] = primitive_ids;
}
```

#### kNN Queries

The general query for k nearest neighbors uses a helper class `tf::nearest_neighbors`. It maintains a sorted range of `k` nearest neighbors, as a view into your buffer. It behaves as a range of size `n`, where `n` is the number of found neighbors (when using a search-radius, `n` might be less than `k`).

```c++
std::array<tf::nearest_neighbor<int, float, 3>, 10> buffer;
auto knn = tf::make_nearest_neighbors(buffer.begin(), k
                                      /*, search_radius*/);
tf::neighbor_search(form, primitive, knn);
for (auto [primitive_id, metric_point] : knn) {}
```

### Ray Casting on Forms

We extend the [Ray Casting on Primitives](#ray-casting) in the following way:

```c++
auto result0 = tf::ray_cast(
                ray, form,
                tf::make_ray_config(min_t, max_t));
if(result0) {
    auto [primitive_id, r_cast] = result0;
}

auto result1 = tf::ray_hit(
                ray, form,
                tf::make_ray_config(min_t, max_t));
if(result1) {
    auto [primitive_id, r_hit] = result0;
}
```
