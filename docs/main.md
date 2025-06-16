# trueform

**High-Performance Geometry Processing**

`trueform` is a geometry processing library built for performance and call-site integration.
It operates directly on your existing data using zero-copy views and inline policy injection — no changes to your architecture required.

For example, you can replace your use of `nanoflann` for *k-NN* queries with just a few lines of code — and gain performance:
```c++
std::vector<float> raw_points;
auto pts = tf::make_point_range<3>(raw_points);
tf::tree<int, float, 3> point_tree(pts, tf::config_tree(4, 4));
auto query_pt = tf::random_point<float, 3>();
std::array<tf::nearest_neighbor<int, float, 3>, 10> knn_buffer;
tf::neighbor_search(tf::make_form( // optional_transformation,
                        point_tree, pts),
                    query_pt,
                    tf::make_nearest_neighbors(knn_buffer.begin(), 10 /*, search_radius*/));
```

<p float="left">
  <img src="./img/nano-build.png" width="49%" />
  <img src="./img/nano-knn.png" width="49%" />
</p>

or replace your use of `CGAL` for *mesh-intersection* queries with an equally minimal call — and significantly faster execution:
```c++
std::vector<int> raw_triangle_ids;
auto triangles = tf::make_polygon_range(
    tf::make_blocked_range<3>(raw_triangle_ids), pts);
tf::tree<int, float, 3> tree(triangles, tf::config_tree(4, 4));
std::vector<std::pair<int, int>> intersecting_primitives;
tf::gather_id_pairs(
    tf::make_form(tree, triangles),
    tf::make_form( // M.dot(x - pts[0]) + pts[10]
        tf::random_frame_at(pts[0], pts[10]), tree, triangles),
    tf::intersects_f, std::back_inserter(intersecting_primitives));
```
<p float="left">
  <img src="./img/cgal-build.png" width="49%" />
  <img src="./img/cgal_speedup.png" width="49%" />
</p>

This reflects `trueform`’s design principles: it’s meant to feel like composing ranges and lambdas — inline, expressive, and non-invasive.

## Quickstart

Add this to your `CMakeLists.txt` to fetch and link `trueform`:

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
