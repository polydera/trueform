#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../util/read_mesh.hpp"
#include "trueform/trueform.hpp"
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangle_3.h>

// --- CGAL Type Definitions ---
using Kernel = CGAL::Simple_cartesian<float>;
using Point_3 = Kernel::Point_3;
using Triangle = Kernel::Triangle_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits = CGAL::AABB_traits<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

// Helper to apply a trueform transformation to a CGAL mesh.
void apply_transform_to_mesh(const Surface_mesh &source_mesh,
                             Surface_mesh &target_mesh,
                             const tf::transformation<float, 3> &transform) {
  for (auto v_idx : target_mesh.vertices()) {
    const auto &cgal_pt = source_mesh.point(v_idx);
    tf::point<float, 3> pt{cgal_pt.x(), cgal_pt.y(), cgal_pt.z()};

    // Apply the transformation
    pt = tf::transformed(pt, transform);

    target_mesh.point(v_idx) = Point_3{pt[0], pt[1], pt[2]};
  }
}

// --- Benchmark Functions ---

int run_intersections(const std::string &file_name, int n_iters) {
  std::cout << "\n--- Running Intersection Benchmark (" << n_iters
            << " iterations) ---\n";
  std::cout << "--- NOTE on methodology: This benchmark compares two slightly "
               "different operations.\n"
            << "--- `trueform` finds all intersecting (mesh0, mesh1) pairs.\n"
            << "--- CGAL's `all_intersected_primitives` collects IDs from "
               "mesh0 that intersect\n"
            << "--- with *any* primitive in mesh1, hence performing "
               "fewer checks.\n"
            << "--- and producing less information.\n";

  // Load CGAL mesh
  Surface_mesh mesh0;
  CGAL::IO::read_polygon_mesh(file_name, mesh0);
  Surface_mesh mesh1 = mesh0; // Create a copy to transform

  // Build CGAL tree once
  Tree tree0(faces(mesh0).begin(), faces(mesh0).end(), mesh0);
  tree0.build();
  tree0.accelerate_distance_queries();

  // Load trueform mesh
  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());
  auto polygons = tf::make_polygons(tf::make_blocked_range<3>(raw_faces),
                                    tf::make_points<3>(raw_points));
  // Build trueform tree once
  tf::aabb_tree<int, float, 3> tree_tf;
  tree_tf.build(polygons, tf::config_tree(4, 4));

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

    // --- CGAL Intersection Test ---
    Tree tree1(faces(mesh1).begin(), faces(mesh1).end(), mesh1);
    tree1.build();

    std::vector<Primitive::Id> cgal_ids;
    tf::tick();
    tree0.all_intersected_primitives(tree1, std::back_inserter(cgal_ids));
    time_cgal += tf::tock();
    dummy_out += cgal_ids.size();

    // --- trueform Intersection Test ---
    std::vector<std::pair<int, int>> tf_id_pairs;

    tf::tick();
    tf::gather_ids(tf::make_form(tree_tf, polygons),
                   tf::make_form(tf::make_frame(transform), tree_tf, polygons),
                   tf::intersects_f, std::back_inserter(tf_id_pairs));
    time_tf += tf::tock();
    dummy_out += tf_id_pairs.size();
  }

  std::cout << "\n\n--- Intersection Results ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Average trueform time: " << (time_tf / n_iters) << " ms\n";
  std::cout << "Average CGAL time:     " << (time_cgal / n_iters) << " ms\n";
  std::cout << "Speed-up:              " << (time_cgal / time_tf) << "x\n";

  return dummy_out;
}

int run_build(const std::string &file_name, int n_iters) {
  std::cout << "\n--- Running Tree Build Benchmark (" << n_iters
            << " iterations) ---\n";

  // Load data
  Surface_mesh mesh;
  CGAL::IO::read_polygon_mesh(file_name, mesh);

  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());
  auto polygons = tf::make_polygons(tf::make_blocked_range<3>(raw_faces),
                                    tf::make_points<3>(raw_points));

  float time_cgal = std::numeric_limits<float>::max();
  float time_tf = std::numeric_limits<float>::max();
  int dummy_out = 0;

  for (int i = 0; i < n_iters; ++i) {
    int progress = static_cast<int>(((i + 1.0) / n_iters) * 100.0);
    std::cout << "\rProgress: " << progress << "%" << std::flush;

    // --- CGAL Build ---
    tf::tick();
    Tree tree_cgal(faces(mesh).begin(), faces(mesh).end(), mesh);
    tree_cgal.build();
    time_cgal = std::min(time_cgal, tf::tock());
    dummy_out += tree_cgal.size();

    // --- trueform Build ---
    tf::tick();
    tf::aabb_tree<int, float, 3> tree_tf;
    tree_tf.build(polygons, tf::config_tree(4, 4));
    time_tf = std::min(time_tf, tf::tock());
    dummy_out += tree_tf.nodes().size();
  }

  std::cout << "\n\n--- Build Results (Best of " << n_iters << ") ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Best trueform time: " << time_tf << " ms\n";
  std::cout << "Best CGAL time:     " << time_cgal << " ms\n";
  std::cout << "Speed-up:           " << (time_cgal / time_tf) << "x\n";

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
  int result2 = run_build(file_path, 10);

  // Return a value based on the computations to ensure the compiler doesn't
  // optimize them away.
  return result1 + result2;
}
