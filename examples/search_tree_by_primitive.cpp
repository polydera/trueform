#include "./util/read_mesh.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/closest_point_on_triangle.hpp"
#include "trueform/intersects.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "trueform/random.hpp"
#include "trueform/search.hpp"
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

  tf::tree<int, float, 3> mesh_tree;
  mesh_tree.build(
      /*tf::strategy::floyd_rivest (or some other strategy),*/
      triangles, tf::config_tree(4, 4));
  std::cout << "---------------------------------" << std::endl;
  std::cout << "Build triangle tree." << std::endl;
  std::cout << "---------------------------------" << std::endl;

  std::cout << "We will pick a random triangle and compute its center. Then we "
               "will find all primitive ids within epsilon of this center."
            << std::endl;

  // pick random triangle
  auto id = tf::random<int>(0, triangles.size() - 1);
  const auto &triangle = triangles[id];
  auto center = tf::make_point((triangle[0].as_vector_view() +
                                triangle[1].as_vector_view() +
                                triangle[2].as_vector_view()) /
                               3);

  // find all triangles within epsilon of center

  std::vector<int> ids_in_tolerance;
  tf::search(
      mesh_tree,
      [&center](const tf::aabb<float, 3> &aabb) {
        return tf::intersects(center, aabb,
                              std::numeric_limits<float>::epsilon());
      },
      [&center, &triangles = triangles,
       &ids_in_tolerance](const auto &triangle_id) {
        if ((center -
             tf::closest_point_on_triangle(triangles[triangle_id], center))
                .length2() < std::numeric_limits<float>::epsilon())
          ids_in_tolerance.push_back(triangle_id);
        // return true (inside condition) if you want to stop the search at
        // first "collision"
        /*return false;*/
      });

  std::cout << "Selected center of triangle with id: " << id << std::endl;
  std::cout << "Found " << ids_in_tolerance.size()
            << " triangles within epsilon of it:" << std::endl;
  for (auto id : ids_in_tolerance)
    std::cout << "  " << id << std::endl;
}
