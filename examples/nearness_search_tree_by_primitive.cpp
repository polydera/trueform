#include "./util/read_mesh.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/closest_point_on_triangle.hpp"
#include "trueform/distance.hpp"
#include "trueform/metric_point.hpp"
#include "trueform/nearest_neighbor.hpp"
#include "trueform/nearness_search.hpp"
#include "trueform/normalized.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "trueform/random_vector.hpp"
#include "trueform/tree.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input.obj>\n";
    return 1;
  }

  std::cout << "Reading file: " << argv[1] << std::endl;
  auto [raw_points, raw_triangle_faces] = tf::examples::read_mesh(argv[1]);
  // create a point range from std::vector<float>
  auto points = tf::make_points<3>(raw_points);
  // create a polygon range from std::vector<int> and points range
  auto triangles = tf::make_polygons(
      tf::make_blocked_range<3>(raw_triangle_faces), points);
  std::cout << "  number of triangles: " << triangles.size() << std::endl;
  std::cout << "  number of points   : " << points.size() << std::endl;
  std::cout << "---------------------------------" << std::endl;

  tf::tree<int, float, 3> mesh_tree;
  mesh_tree.build(
      /*tf::strategy::floyd_rivest (or some other strategy),*/
      triangles, tf::config_tree(4, 4));
  std::cout << "Build triangle tree." << std::endl;
  std::cout << "---------------------------------" << std::endl;

  std::cout << "We will pick a random point on the sphere enclosing the mesh "
               "and compute the closest point on the mesh."
            << std::endl;
  std::cout << "---------------------------------" << std::endl;

  auto center = mesh_tree.nodes().front().aabb.center();
  auto radius = mesh_tree.nodes().front().aabb.diagonal().length() / 2;
  auto query_pt =
      center + tf::normalized(tf::random_vector<float, 3>()) * radius;

  auto [primitive_id, closest_point] = tf::nearness_search(
      mesh_tree,
      [query_pt](const tf::aabb<float, 3> &aabb) {
        return tf::distance2(aabb, query_pt);
      },
      [&query_pt, &triangles = triangles](const auto &triangle_id) {
        auto cpt =
            tf::closest_point_on_triangle(triangles[triangle_id], query_pt);
        return tf::make_metric_point((cpt - query_pt).length2(), cpt);
      } /*, search_radius */);

  auto [metric, point] = closest_point;

  std::cout << "Closest point: " << point[0] << ", " << point[1] << ", "
            << point[2] << " on primitive: " << primitive_id << std::endl;
  std::cout << "At distance: " << tf::sqrt(metric) << " from query_pt"
            << std::endl;

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Now we will compute 4 nearest points" << std::endl;
  std::cout << "(If the closest point is on a vertex, these might be the same)"
            << std::endl;

  std::array<tf::nearest_neighbor<int, float, 3>, 4> closest_points;
  auto knn = tf::make_nearest_neighbors(closest_points.begin(),
                                        4 /*, search_radius */);

  tf::nearness_search(
      mesh_tree,
      [query_pt](const tf::aabb<float, 3> &aabb) {
        return tf::distance2(aabb, query_pt);
      },
      [&query_pt, &triangles = triangles](const auto &triangle_id) {
        auto cpt =
            tf::closest_point_on_triangle(triangles[triangle_id], query_pt);
        return tf::make_metric_point((cpt - query_pt).length2(), cpt);
      },
      knn);

  std::cout << "---------------------------------" << std::endl;
  std::cout << "Closest points: " << knn.size() << std::endl;
  for (auto [primitive_id, closest_point] : knn) {
    auto [metric, point] = closest_point;
    std::cout << "  Closest point: " << point[0] << ", " << point[1] << ", "
              << point[2] << " on primitive: " << primitive_id << std::endl;
    std::cout << "  At distance: " << tf::sqrt(metric) << " from query_pt"
              << std::endl;
  }
}
