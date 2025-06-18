#include "./util/read_mesh.hpp"
#include "trueform/blocked_range.hpp"
#include "trueform/points.hpp"
#include "trueform/polygons.hpp"
#include "trueform/tree.hpp"
#include "trueform/tree_config.hpp"
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

  // alternatively, you can specify how to compute the aabb
  // inside the tf::config_tree
  // tf::config_tree(4, 4, [](const auto &obj) {
  //  return tf::aabb_from(obj);
  //})

  tf::tree<int, float, 3> point_tree;
  point_tree.build(
      /*tf::strategy::floyd_rivest (or some other strategy),*/
      points, tf::config_tree(4, 4));
  std::cout << "---------------------------------" << std::endl;
  std::cout << "Build point tree." << std::endl;

  return mesh_tree.nodes().size() + point_tree.nodes().size();
}
