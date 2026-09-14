#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/csg/csg_created_points.hpp>
#include <trueform/cpp/csg/csg_graph.hpp>
#include <trueform/cpp/csg/csg_mesh.hpp>
#include <trueform/cpp/csg/outer_shell.hpp>
#include <trueform/cpp/geometry/make_box_mesh.hpp>

#include <iostream>
#include <utility>
#include <vector>

int main() {
  using real = double;
  using index = tf::cpp::default_index_t;
  using mesh_type = tf::cpp::mesh<index, real>;

  const auto left = tf::cpp::make_box_mesh(real{2}, real{2}, real{2});
  auto right_storage = tf::cpp::make_box_mesh(real{2}, real{2}, real{2});

  // Move the second box in its own coordinates, before anything is asked of
  // the cache that will remember it. A caller that moves points AFTER a read
  // says so: `cache.points_changed()`.
  for (auto point : right_storage.points())
    point[0] += real{0.75};

  tf::cpp::cache<index, real> left_cache;
  tf::cpp::cache<index, real> right_cache;

  // The graph reads its operands for as long as it lives, and an operand is a
  // reading of storage and a cache this scope holds: both outlive the graph.
  std::vector<mesh_type> operands;
  operands.push_back({left.faces(), left.points(), left_cache});
  operands.push_back(
      {right_storage.faces(), right_storage.points(), right_cache});
  const auto graph = tf::cpp::make_csg_graph(std::move(operands));

  // One arrangement, arbitrarily many expressions answered against it.
  const auto united =
      tf::cpp::make_csg_mesh(graph, tf::csg::op(0) | tf::csg::op(1));
  const auto difference =
      tf::cpp::make_csg_mesh(graph, tf::csg::op(0) - tf::csg::op(1));
  const auto overlap =
      tf::cpp::make_csg_mesh(graph, tf::csg::op(0) & tf::csg::op(1));
  const auto created_points = tf::cpp::csg_created_points(graph);

  // A result is storage like any other: give it a cache and it is a mesh.
  tf::cpp::cache<index, real> united_cache;
  const mesh_type united_mesh{united.faces(), united.points(), united_cache};
  const auto shell = tf::cpp::outer_shell(united_mesh);

  std::cout << "one CSG graph, three extractions:\n"
            << "  union:        " << united.size() << " triangles\n"
            << "  difference:   " << difference.size() << " triangles\n"
            << "  intersection: " << overlap.size() << " triangles\n"
            << "  created intersection vertices: " << created_points.shape_at(0)
            << '\n'
            << "  repaired outer shell: " << shell.size() << " triangles\n";

  return graph.is_valid() && united.size() > 0 && difference.size() > 0 &&
                 overlap.size() > 0 && created_points.shape_at(0) > 0 &&
                 shell.size() > 0
             ? 0
             : 1;
}
