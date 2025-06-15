#include "./util/read_mesh.hpp"
#include "trueform/aabb_metrics.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/metric_point_pair.hpp"
#include "trueform/nearness_search.hpp"
#include "trueform/normalized.hpp"
#include "trueform/point_range.hpp"
#include "trueform/polygon_range.hpp"
#include "trueform/random.hpp"
#include "trueform/random_transformation.hpp"
#include "trueform/random_vector.hpp"
#include "trueform/sqrt.hpp"
#include "trueform/transformation.hpp"
#include "trueform/transformed.hpp"
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
  auto points = tf::make_point_range<3>(raw_points);
  // create a polygon range from std::vector<int> and points range
  auto triangles = tf::make_polygon_range(
      tf::make_blocked_range<3>(raw_triangle_faces), points);
  std::cout << "  number of triangles: " << triangles.size() << std::endl;
  std::cout << "  number of points   : " << points.size() << std::endl;
  std::cout << "---------------------------------" << std::endl;

  // pick random point
  auto id0 = tf::random<int>(0, points.size() - 1);
  auto pt0 = points[id0];
  auto id1 = tf::random<int>(0, points.size() - 1);
  auto pt1 = points[id1];

  tf::tree<int, float, 3> tree;
  tree.build(
      /*tf::strategy::floyd_rivest,*/
      points, tf::config_tree(4, 4));
  std::cout << "Build point tree." << std::endl;
  std::cout << "---------------------------------" << std::endl;

  std::cout << "We will use the points of the dataset. We will transform one "
               "copy of the point-cloud so that two ids are at epsilon "
               "distance to eachother. Then we will compute the closest point "
               "between the two clouds."
            << std::endl;
  std::cout << "---------------------------------" << std::endl;

  // create transform that:
  // 1. makes pt1 the origin
  // 2. applies a random rotation
  // 3. aligns pt1 to epsilon displaced pt0
  auto dpt0 = pt0 + tf::normalized(tf::random_vector<float, 3>()) * 1.e-7f;
  auto transformation = tf::transformed(
      tf::make_transformation_from_translation(-pt1.as_vector_view()),
      tf::random_transformation<float>(dpt0.as_vector_view()));

  // apply the transformation to the aabbs and primitives.
  auto [primitive_ids, closest_point_pair] = tf::nearness_search(
      tree, tree,
      [&](const auto &aabb0, const auto &aabb1) { // transform the aabb
        return tf::make_aabb_metrics(aabb0,
                                     tf::transformed(aabb1, transformation));
      },
      [&points = points, &transformation](auto id0, auto id1) {
        auto tpt = tf::transformed(points[id1], transformation);
        auto d2 = (points[id0] - tpt).length2();
        return tf::make_metric_point_pair(d2, points[id0], tpt);
      } /*, search_radius */);

  auto [primitive_id0, primitive_id1] = primitive_ids;
  auto [metric, point0, point1] = closest_point_pair;

  std::cout << "Closest points are on primitives: " << primitive_id0 << ", "
            << primitive_id1 << std::endl;
  std::cout << "At distance: " << tf::sqrt(metric) << std::endl;
}
