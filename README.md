# trueform

**Real-time geometry built on composable range-based policies**

`trueform` is a C++ library for real-time geometric processing, built on the principles of composable views and inline policy injection. It operates directly on you *plain-old-data*, by providing semantic views that wrap it with geometric meaning. From individual primitives to structured ranges, from meta-data injection to spatial queries, every operation happens directly on your data; enriched with semantics without architectural changes.

The library integrates directly at the call site: no boilerplate, no architectural rewrites, no heavyweight setup. It acts as a lightweight, expressive layer over your existing data. Like C++ ranges or lambdas, it lets you build rich, semantic geometry inline, without sacrificing performance or control.

### Simplifying Common Tasks

`trueform` can serve as a simpler, faster replacement for common operations you already perform with other libraries. For example, you can replace `nanoflann` for *k-NN* queries with just a few lines of code:
```c++
std::vector<float> raw_points;
auto pts = tf::make_points<3>(raw_points);
tf::tree<int, float, 3> point_tree(pts, tf::config_tree(4, 4));
auto query_pt = tf::random_point<float, 3>();
std::array<tf::nearest_neighbor<int, float, 3>, 10> knn_buffer;
auto knn = tf::neighbor_search(
    tf::make_form( // optional_transformation,
        point_tree, pts),
    query_pt,
    tf::make_nearest_neighbors(knn_buffer.begin(), 10 /*, search_radius*/));
for (auto [primitive_id, metric_point] : knn) {
    auto [distance2, closest_point] = metric_point;
}
```

<p float="left">
  <img src="./docs/img/nano-build.png" width="49%" />
  <img src="./docs/img/nano-knn.png" width="49%" />
</p>

or replace your use of `CGAL` for *mesh-intersection* queries with an equally minimal call — and significantly faster execution:
```c++
std::vector<int> raw_triangle_ids;
auto triangles = tf::make_polygons(
    tf::make_blocked_range<3>(raw_triangle_ids), pts);
tf::tree<int, float, 3> tree(triangles, tf::config_tree(4, 4));
std::vector<std::pair<int, int>> intersecting_primitives;
tf::gather_ids(
    tf::make_form(tree, triangles),
    tf::make_form( // M.dot(x - pts[0]) + pts[10]
        tf::random_frame_at(pts[0], pts[10]), tree, triangles),
    tf::intersects_f, std::back_inserter(intersecting_primitives));
```
<p float="left">
  <img src="./docs/img/cgal-build.png" width="49%" />
  <img src="./docs/img/cgal_speedup.png" width="49%" />
</p>

This reflects `trueform`’s design principles: it’s meant to feel like composing ranges and lambdas — inline, expressive, and non-invasive. 

### Showcase: The Power of Composition

While `trueform` can replace existing tools, its real power is its expressive, compositional API. Consider a common challenge: modeling a moving point cloud of "emitters." Each emitter needs a unique ID, a static mounting normal, a dynamic aiming direction, and a color—all sourced from different data vectors. With `trueform`, this complex assembly becomes a single, readable pipeline:
```c++
// --- Raw Data ---
std::vector<float> raw_pts;
std::vector<float> raw_normals;
std::vector<std::string> ids;
std::vector<float> raw_direction;
std::vector<std::array<float, 3>> colors;

// --- Composition Pipeline ---
// Start with raw points and progressively enrich them with semantics.
auto emitters =
    tf::make_points<3>(raw_pts)
    // 1. Tag the entire collection with a single ID.
    | tf::tag_id("emitter_array_alpha")
    // 2. Zip per-element data onto the range.
    | tf::zip_ids(ids)
    | tf::zip_normals(
          tf::make_unit_vectors<3>(raw_normals)
          // Policies can be nested: tag the zipped normals themselves.
          | tf::tag_id("mount_normals"))
    // 3. Zip a composite state from multiple data sources.
    | tf::zip_states(
          tf::make_unit_vectors<3>(raw_direction)
          | tf::tag_id("aim_directions"),
          tf::make_view(colors)
          | tf::tag_id("color_data"));
```
The `emitters` object still behaves like a simple range of points, compatible with any algorithm on point primitives. The real power is realized in the query, where its attached policies are correctly preserved or transformed, enabling complex, stateful logic:
```c++
// --- Using the Enriched Object in a Query ---
tf::tree<int, float, 3> emitter_tree{emitters, tf::config_tree(4, 4)};
tf::frame<float, 3> frame = tf::random_transformation<float, 3>();
auto query_pt = tf::random_point<float, 3>() | tf::tag_id("target");

float aim_radius2 = 4.0f; // Use squared distance for performance
float aim_cos_angle = 0.95f;

tf::search(
    // The form applies the dynamic frame to the static tree and emitters.
    tf::make_form(frame, emitter_tree, emitters),
    // Broad-phase: Quickly cull any nodes outside the target radius.
    [&](const auto &aabb) {
        return tf::distance2(aabb, query_pt) < aim_radius2;
    },
    // Narrow-phase: The "brain" of the query. Evaluate each potential
    // emitter.
    // auto transformed_emitter = tf::transformed(emitters[index], frame);
    [&](const auto &transformed_emitter) {
        if (tf::distance2(transformed_emitter, query_pt) >= aim_radius2)
        return;
        // Access all the policy data, which has been correctly handled.
        const auto &id =
            transformed_emitter.id(); // The per-emitter ID is preserved.
        const auto &mount_normal =
            transformed_emitter.normal(); // The normal vector is transformed.
        // `direction` was transformed, while `color` was passed through
        // unchanged.
        const auto &[aim_direction, color] = transformed_emitter.state();
        auto to_target = tf::normalized(query_pt - transformed_emitter);
        // Check if the target is in the turret's firing arc and aimed
        // correctly.
        if (tf::dot(to_target, mount_normal) > 0 &&
            tf::dot(to_target, aim_direction) > aim_cos_angle) {

        std::cout << "Emitter " << id << " can hit " << query_pt.id()
                    << "!\n";
        }
    });
```
The result is a highly expressive, architecture-agnostic approach to geometry that integrates into your existing code.

## Getting Started

### Requirements

* **C++17** or later
* **Intel TBB** (Threading Building Blocks) for parallelism

### Installation

`trueform` is a header-only library. The recommended way to integrate it into your project is with CMake's `FetchContent`.

Add the following to your `CMakeLists.txt`:

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
We provide two main resources to help you get started and master trueform:

### [Tutorial](./docs/tutorial.md)

**This is the best place to start.** It's a comprehensive document that serves as both a guided tour of the library's philosophy and a complete reference manual for the `core`, `spatial`, and `topology` modules.

### [Examples](./examples/)

For hands-on, practical applications, explore the examples directory. It contains a collection of standalone programs organized into three categories:
* **Core Functionality**: Self-contained demonstrations of primary features.
* **Comparisons**: Performance and usage comparisons against libraries like `nanoflann`.
* **VTK Integration**: Guides for using trueform with tools like `VTK`.

## Publications

- > **tf::tree: A General-Purpose Spatial Hierarchy for Real-Time Geometry Queries.**  
Žiga Sajovic, Dejan Knez, and Robert Korez.  
*Institute of Electrical and Electronics Engineers (IEEE),* June 2025.  
[https://doi.org/10.36227/techrxiv.174952959.92977743/v1](https://doi.org/10.36227/techrxiv.174952959.92977743/v1)

- > **Real-Time Mesh Booleans that Commute with Mesh Idealization.**  
Žiga Sajovic and Dejan Knez  
 *Institute of Electrical and Electronics Engineers (IEEE),* May 2025.  
 [https://doi.org/10.36227/techrxiv.174667714.42575478/v1](https://doi.org/10.36227/techrxiv.174667714.42575478/v1)
