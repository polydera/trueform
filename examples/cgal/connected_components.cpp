#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <iostream>

#include "../util/read_mesh.hpp"
#include "trueform/trueform.hpp"
#include <filesystem>

namespace PMP = CGAL::Polygon_mesh_processing;
using Kernel = CGAL::Simple_cartesian<double>;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;

int dummy = 0;

float run_cgal(std::string file_name, int n_iters) {

  Mesh mesh0;
  CGAL::IO::read_polygon_mesh(file_name, mesh0);

  float time = std::numeric_limits<float>::max();
  for (int i = 0; i < n_iters; ++i) {
    Mesh mesh = mesh0;

    // Prepare a property map for storing face labels
    using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;
    Mesh::Property_map<face_descriptor, std::size_t> fccmap;
    fccmap = mesh.add_property_map<face_descriptor, std::size_t>("f:CC").first;

    // Compute connected components
    tf::tick();
    std::size_t num = PMP::connected_components(
        mesh, fccmap,
        PMP::parameters::all_default()); // Optional constraints like
                                         // edge_is_constrained
    time = std::min(time, tf::tock());
    dummy += num;
  }
  return time;
}

float run_tf(std::string file_name, int n_iters) {
  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());

  auto faces = tf::make_blocked_range<3>(raw_faces);
  auto pts = tf::make_points<3>(raw_points);
  auto polygons = tf::make_polygons(faces, pts);
  tf::face_membership<int> fm;
  fm.build(polygons);
  tf::manifold_edge_link<int, 3> mel;
  mel.build(faces, fm);
  auto applier = [&](int id, auto &&f) {
    for (const auto &h : mel[id])
      if (h.is_simple())
        f(h.face_peer);
  };

  float time = std::numeric_limits<float>::max();
  for (int i = 0; i < n_iters; ++i) {
    tf::buffer<int> labels;
    labels.allocate(faces.size());
    tf::tick();
    auto n_components = tf::label_connected_components<int>(labels, applier);
    time = std::min(time, tf::tock());
    dummy += n_components;
  }
  return time;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input.obj>\n";
    return 1;
  }

  std::string file_path = argv[1];
  if (!std::filesystem::exists(file_path)) {
    std::cerr << "Error: File not found at " << file_path << std::endl;
    return 1;
  }

  std::cout << "Benchmarking with file: " << file_path << std::endl;
  std::cout << "==================================================\n";

  auto time_tf = run_tf(file_path, 20);
  auto time_cgal = run_cgal(file_path, 20);

  std::cout << " tf time: " << time_tf << " ms" << std::endl;
  std::cout << " cgal time: " << time_cgal << " ms" << std::endl;
  std::cout << "  speed-up: " << (time_cgal / time_tf) << std::endl;
  return dummy;
}
