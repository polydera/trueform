#include "./util/read_mesh.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/intersects.hpp"
#include "trueform/local_vector.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "trueform/random.hpp"
#include "trueform/random_transformation.hpp"
#include "trueform/search.hpp"
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
  auto points = tf::make_points<3>(raw_points);
  // create a polygon range from std::vector<int> and points range
  auto triangles = tf::make_polygons(
      tf::make_blocked_range<3>(raw_triangle_faces), points);
  std::cout << "  number of triangles: " << triangles.size() << std::endl;
  std::cout << "  number of points   : " << points.size() << std::endl;
  std::cout << "---------------------------------" << std::endl;

  // pick random point
  auto id0 = tf::random<int>(0, points.size() - 1);
  auto pt0 = points[id0];
  auto id1 = tf::random<int>(0, points.size() - 1);
  auto pt1 = points[id1];

  std::cout << "We will use the points of the dataset. We will place spheres "
               "with radius epsilon on each point and "
               "transform one copy of the point-cloud. Then we will find all "
               "intersecting sphere pairs."
            << std::endl;
  std::cout << "Selected points with ids: " << id0 << ", " << id1
            << " to align under random rotation." << std::endl;

  tf::tree<int, float, 3> tree;
  tree.build(tf::strategy::floyd_rivest, points, tf::config_tree(4, 4));
  std::cout << "---------------------------------" << std::endl;
  std::cout << "Build point tree." << std::endl;
  std::cout << "---------------------------------" << std::endl;

  // create transform that:
  // 1. makes pt1 the origin
  // 2. applies a random rotation
  // 3. aligns pt1 to pt0
  auto transformation =
      tf::transformed(tf::make_transformation_from_translation(-pt1.as_vector_view()),
                      tf::random_transformation<float>(pt0.as_vector_view()));

  tf::local_vector<std::pair<int, int>> local_ids;
  // we may use the same tree, as we will simply
  // apply the transformation to the aabbs and primitives.
  tf::search(
      tree, tree,
      [&](const auto &aabb0, const auto &aabb1) { // transform the aabb
        return tf::intersects(aabb0, tf::transformed(aabb1, transformation),
                              std::numeric_limits<float>::epsilon());
      },
      [&points = points, &transformation, &local_ids](auto id0, auto id1) {
        if ((points[id0] - tf::transformed(points[id1], transformation))
                .length2() < std::numeric_limits<float>::epsilon())
          local_ids.emplace_back(id0, id1);
        // return true (inside condition) if you want to stop the search at
        // first "collision"
        /*return true;*/
      });

  auto ids = local_ids.to_vector();
  std::cout << "Found " << ids.size()
            << " point pairs within epsilon of eachother" << std::endl;
  for (auto id : ids)
    std::cout << "  " << id.first << ", " << id.second << std::endl;
}
