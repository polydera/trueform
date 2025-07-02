#include "../util/read_mesh.hpp"
#include "trueform/trueform.hpp"
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangle_3.h>

using Kernel = CGAL::Simple_cartesian<float>;
using Point_3 = Kernel::Point_3;
using Triangle = Kernel::Triangle_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits = CGAL::AABB_traits<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

auto random_intersected_configuration(
    const Surface_mesh &in_mesh, Surface_mesh &mesh,
    const tf::transformation<float, 3> transform) {
  for (auto v : mesh.vertices()) {
    auto ppt = in_mesh.point(v);
    tf::point<float, 3> pt{{ppt[0], ppt[1], ppt[2]}};
    pt = transform.transform_point(pt);
    mesh.point(v) = Point_3{pt[0], pt[1], pt[2]};
  }
}

int run_intersections(std::string file_name, int n_iters) {
  int out = 0;
  Surface_mesh mesh0;
  CGAL::IO::read_polygon_mesh(file_name, mesh0);
  Surface_mesh mesh1;
  CGAL::IO::read_polygon_mesh(file_name, mesh1);

  Tree tree0(faces(mesh0).begin(), faces(mesh0).end(), mesh0);
  tree0.build();
  tree0.accelerate_distance_queries();

  auto [raw_points, raw_triangle_faces] =
      tf::examples::read_mesh(file_name.c_str());
  tf::tree<int, float, 3> tree;
  auto polygons =
      tf::make_polygons(tf::make_blocked_range<3>(raw_triangle_faces),
                        tf::make_points<3>(raw_points));
  tree.build(polygons, tf::config_tree(4, 4));

  float time_cgal = 0;
  float time_tf = 0;

  for (int i = 0; i < n_iters; ++i) {

    auto id0 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt0 = polygons.points()[id0];
    auto id1 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt1 = polygons.points()[id1];
    auto transformation = tf::transformed(
        tf::make_transformation_from_translation(-pt1.as_vector_view()),
        tf::random_transformation(pt0.as_vector_view()));
    random_intersected_configuration(mesh0, mesh1, transformation);

    Tree tree1(faces(mesh1).begin(), faces(mesh1).end(), mesh1);
    tree1.build();

    std::vector<int> ids;
    tf::tick();
    tree0.all_intersected_primitives(tree1, std::back_inserter(ids));
    time_cgal += tf::tock();
    out += ids.size();

    std::vector<std::pair<int, int>> id_pairs;
    tf::tick();
    tf::gather_ids(
        tf::make_form(tree, polygons),
        tf::make_form(tf::make_frame(transformation), tree, polygons),
        tf::intersects_f, std::back_inserter(id_pairs));
    time_tf += tf::tock();

    out += id_pairs.size();

    std::cout << "tf: " << (time_tf / (i + 1)) << std::endl;
    std::cout << "cgal: " << (time_cgal / (i + 1)) << std::endl;
    std::cout << "speedup: " << (time_cgal / time_tf) << std::endl;
  }
  std::cout << "tf: " << (time_tf / n_iters) << std::endl;
  std::cout << "cgal: " << (time_cgal / n_iters) << std::endl;
  std::cout << "speedup: " << (time_cgal / time_tf) << std::endl;
  return out;
}

int run_build(std::string file_name, int n_iters) {
  int out = 0;
  Surface_mesh mesh0;
  CGAL::IO::read_polygon_mesh(file_name, mesh0);
  Surface_mesh mesh1;
  CGAL::IO::read_polygon_mesh(file_name, mesh1);
  auto [raw_points, raw_triangle_faces] =
      tf::examples::read_mesh(file_name.c_str());
  tf::tree<int, float, 3> tree;
  auto polygons =
      tf::make_polygons(tf::make_blocked_range<3>(raw_triangle_faces),
                        tf::make_points<3>(raw_points));

  float time_cgal = std::numeric_limits<float>::max();
  float time_tf = std::numeric_limits<float>::max();

  for (int i = 0; i < n_iters; ++i) {
    tf::tick();
    Tree tree0(faces(mesh0).begin(), faces(mesh0).end(), mesh0);
    tree0.build();
    time_cgal = std::min(time_cgal, tf::tock());
    out += tree0.size();

    tf::tick();
    tree.build(polygons, tf::config_tree(4, 4));
    time_tf = std::min(time_tf, tf::tock());
    out += tree.nodes().size();
  }

  std::cout << "tf: " << (time_tf) << std::endl;
  std::cout << "cgal: " << (time_cgal) << std::endl;
  std::cout << "speedup: " << (time_cgal / time_tf) << std::endl;
  return out;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input.obj>\n";
    return 1;
  }

  return run_intersections(argv[1], 10) + run_build(argv[1], 10);
}
