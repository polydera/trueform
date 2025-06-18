#include "./util/read_mesh.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/intersects.hpp"
#include "trueform/local_vector.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "trueform/random.hpp"
#include "trueform/random_vector.hpp"
#include "trueform/search_self.hpp"
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

  std::cout
      << "We will use the points of this dataset to perform a "
         "self-search for all primitive pairs in epsilon proximity of "
         "eachother.This is an example of self - intersections under tolerance "
      << std::endl;

  std::cout << "We will select 10 points to duplicate and shift by epsilon in "
               "a random direction."
            << std::endl;

  std::vector<tf::point<float, 3>> duplicated_points(points.begin(),
                                                      points.end());
  float epsilon = std::numeric_limits<float>::epsilon() * 2;
  for (int i = 0; i < 10; ++i) {
    auto pt = points[tf::random<int>(0, points.size() - 1)];
    auto mv = tf::random_vector<float, 3>();
    mv /= mv.length();
    auto dpt = pt + mv * 0.8f * epsilon;
    duplicated_points.push_back(dpt);
  }

  tf::tree<int, float, 3> tree;
  tree.build(
      /*tf::strategy::floyd_rivest, */
      duplicated_points, tf::config_tree(4, 4));
  std::cout << "---------------------------------" << std::endl;
  std::cout << "Build point tree." << std::endl;
  std::cout << "---------------------------------" << std::endl;

  tf::local_vector<std::pair<int, int>> local_ids;
  tf::search_self(
      tree,
      [&](const auto &aabb0, const auto &aabb1) {
        return tf::intersects(aabb0, aabb1, epsilon);
      },
      [&duplicated_points = duplicated_points, &local_ids,
       epsilon2 = epsilon * epsilon](auto id0, auto id1) {
        if ((duplicated_points[id0] - duplicated_points[id1]).length2() <
            epsilon2)
          local_ids.emplace_back(id0, id1);

        // do not stop the local search. Return true if you want to stop on
        // first collision
        /*return true;*/
      });

  auto ids = local_ids.to_vector();
  std::cout << "Found " << ids.size()
            << " point pairs within epsilon of eachother" << std::endl;
  for (auto id : ids)
    std::cout << "  " << id.first << ", " << id.second << std::endl;
  auto ids2_total = local_ids.to_vector();
}
