#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../util/read_mesh.hpp"
#include "trueform/trueform.hpp"
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/polygon_mesh_processing.h>
#include <vector>

// --- CGAL Type Definitions ---
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Polyline = std::vector<Point_3>;

// Helper to apply a trueform transformation to a CGAL mesh.
void apply_transform_to_mesh(const Surface_mesh &source_mesh,
                             Surface_mesh &target_mesh,
                             const tf::transformation<float, 3> &transform) {
  for (auto v_idx : target_mesh.vertices()) {
    const auto &cgal_pt = source_mesh.point(v_idx);
    tf::point<double, 3> pt{cgal_pt.x(), cgal_pt.y(), cgal_pt.z()};

    // Apply the transformation
    pt = tf::transformed(pt, transform);

    target_mesh.point(v_idx) = Point_3{pt[0], pt[1], pt[2]};
  }
}

// --- Benchmark Functions ---

int run_intersections(const std::string &file_name, int n_iters) {
  std::cout << "\n--- Running Intersection Curve Benchmark (" << n_iters
            << " iterations) ---\n";
  // Load CGAL mesh
  Surface_mesh mesh0;
  CGAL::IO::read_polygon_mesh(file_name, mesh0);
  Surface_mesh mesh1 = mesh0; // Create a copy to transform

  // Load trueform mesh
  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());
  auto polygons =
      tf::make_polygons(tf::make_blocked_range<3>(raw_faces),
                        tf::make_points<3>(raw_points).as<double>());
  // Build trueform tree once
  tf::tree<int, float, 3> tree_tf;
  tree_tf.build(polygons, tf::config_tree(4, 4));
  tf::face_membership<int> fe;
  fe.build(polygons);
  tf::manifold_edge_link<int, 3> mel;
  mel.build(polygons.faces(), fe);

  float time_cgal = 0.0f;
  float time_tf = 0.0f;
  int dummy_out = 0; // To prevent compiler from optimizing away results

  for (int i = 0; i < n_iters; ++i) {
    // Print progress indicator that updates on the same line
    int progress = static_cast<int>(((i + 1.0) / n_iters) * 100.0);
    std::cout << "\rProgress: " << progress << "%" << std::flush;

    // Generate a new random transformation for each iteration
    // guaranteeing intersections
    auto id0 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt0 = polygons.points()[id0];
    auto id1 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt1 = polygons.points()[id1];
    auto transform = tf::transformed(
        tf::make_transformation_from_translation(-pt1.as_vector_view()),
        tf::random_transformation(pt0.as_vector_view()));
    apply_transform_to_mesh(mesh0, mesh1, transform);

    // 2. Prepare the output container
    std::vector<Polyline> intersection_polylines;

    tf::tick();
    CGAL::Polygon_mesh_processing::surface_intersection(
        mesh0, mesh1, std::back_inserter(intersection_polylines));
    time_cgal += tf::tock();
    auto form0 = tf::make_form(tf::make_frame(transform), tree_tf,
                               polygons | tf::tag(fe) | tf::tag(mel));
    auto form1 = tf::make_form(tree_tf, polygons | tf::tag(fe) | tf::tag(mel));
    tf::tick();
    tf::intersections_between_polygons<int, double, 3> fi;
    fi.build(form0, form1);
    auto edges = tf::make_intersection_edges(fi);
    time_tf += tf::tock();
    dummy_out += edges.size();
    dummy_out += intersection_polylines.size();
  }

  std::cout << "\n\n--- Intersection Curve Results ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Average trueform time: " << (time_tf / n_iters) << " ms\n";
  std::cout << "Average CGAL time:     " << (time_cgal / n_iters) << " ms\n";
  std::cout << "Speed-up:              " << (time_cgal / time_tf) << "x\n";

  return dummy_out;
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

  int result1 = run_intersections(file_path, 10);
  std::cout << "\n==================================================\n";

  // Return a value based on the computations to ensure the compiler doesn't
  // optimize them away.
  return result1;
}
