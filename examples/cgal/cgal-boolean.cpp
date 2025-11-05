#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../util/read_mesh.hpp"
#include "trueform/trueform.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/repair.h>

using Kernel       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3      = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;

// Helper: apply a trueform transformation to a CGAL mesh (writes into target).
void apply_transform_to_mesh(const Surface_mesh &source_mesh,
                             Surface_mesh &target_mesh,
                             const tf::transformation<float, 3> &transform) {
  for (auto v_idx : target_mesh.vertices()) {
    const auto &cgal_pt = source_mesh.point(v_idx);
    tf::point<double, 3> pt{cgal_pt.x(), cgal_pt.y(), cgal_pt.z()};
    pt = tf::transformed(pt, transform);
    target_mesh.point(v_idx) = Point_3{pt[0], pt[1], pt[2]};
  }
}

int run_boolean_merge_benchmark(const std::string &file_name, int n_iters) {
  std::cout << "\n--- Running Boolean Merge Benchmark (" << n_iters
            << " iterations) ---\n";

  // --- Load CGAL mesh (mesh0), copy to mesh1 (to be transformed) ---
  Surface_mesh mesh0;
  if (!CGAL::IO::read_polygon_mesh(file_name, mesh0) || CGAL::is_empty(mesh0)) {
    std::cerr << "CGAL failed to read mesh or mesh is empty.\n";
    return 0;
  }
  Surface_mesh mesh1 = mesh0;

  // (Optional but wise) Ensure meshes are valid/manifold-ish for CGAL corefinement
  // CGAL::Polygon_mesh_processing::stitch_borders(mesh0);
  // CGAL::Polygon_mesh_processing::stitch_borders(mesh1);

  // --- Load trueform mesh (same file) ---
  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());
  auto polygons =
      tf::make_polygons(tf::make_blocked_range<3>(raw_faces),
                        tf::make_points<3>(raw_points).as<double>());

  // Build trueform acceleration & tags once
  tf::tree<int, float, 3> tree_tf;
  tree_tf.build(polygons, tf::config_tree(4, 4));

  tf::face_membership<int> fe;
  fe.build(polygons);

  tf::manifold_edge_link<int, 3> mel;
  mel.build(polygons.faces(), fe);

  float time_cgal_ms = 0.0f;
  float time_tf_ms   = 0.0f;
  std::size_t dummy_out = 0; // prevent over-optimization

  for (int i = 0; i < n_iters; ++i) {
    int progress = static_cast<int>(((i + 1.0) / n_iters) * 100.0);
    std::cout << "\rProgress: " << progress << "%" << std::flush;

    // --- Random rigid transform that guarantees overlap (same recipe you used) ---
    auto id0 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt0 = polygons.points()[id0];
    auto id1 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt1 = polygons.points()[id1];

    auto transform = tf::transformed(
        tf::make_transformation_from_translation(-pt1.as_vector_view()),
        tf::random_transformation(pt0.as_vector_view()));

    // Apply transform to mesh1 (CGAL side)
    apply_transform_to_mesh(mesh0, mesh1, transform);

    // -------------------
    // CGAL: boolean union
    // -------------------
    Surface_mesh out_union;
    tf::tick();
    // corefine modifies input meshes; pass copies if you need originals preserved beyond this scope
    Surface_mesh a = mesh0;
    Surface_mesh b = mesh1;
    // (Optional) PMP repair/stitching for robustness:
    // CGAL::Polygon_mesh_processing::stitch_borders(a);
    // CGAL::Polygon_mesh_processing::stitch_borders(b);

    bool ok_union = CGAL::Polygon_mesh_processing::corefine_and_compute_union(
        a, b, out_union);

    time_cgal_ms += tf::tock();
    dummy_out += ok_union ? static_cast<std::size_t>(out_union.number_of_faces())
                          : 0;

    // ------------------------
    // trueform: boolean merge
    // ------------------------
    // Build forms: form0 is transformed, form1 is original.
    auto form0 = tf::make_form(tf::make_frame(transform), tree_tf,
                               polygons | tf::tag(fe) | tf::tag(mel));
    auto form1 = tf::make_form(tree_tf, polygons | tf::tag(fe) | tf::tag(mel));

    tf::tick();
    // returns: result mesh, per-face labels, and intersection curves when tf::return_curves is passed
    auto [result_mesh, labels, curves] =
        tf::make_boolean(form0, form1, tf::boolean_op::merge, tf::return_curves);
    time_tf_ms += tf::tock();

    // Touch the results so the compiler can’t elide work
    dummy_out += result_mesh.faces().size();
    dummy_out += labels.size();
    dummy_out += curves.size();
  }

  std::cout << "\n\n--- Boolean Merge Results ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Average trueform time: " << (time_tf_ms / n_iters)   << " ms\n";
  std::cout << "Average CGAL time:     " << (time_cgal_ms / n_iters) << " ms\n";
  if (time_tf_ms > 0.0f)
    std::cout << "Speed-up:              " << (time_cgal_ms / time_tf_ms) << "x\n";
  else
    std::cout << "Speed-up:              n/a (trueform time was ~0)\n";

  return static_cast<int>(dummy_out & 0x7fffffff);
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

  int result = run_boolean_merge_benchmark(file_path, /*n_iters=*/10);
  std::cout << "\n==================================================\n";
  return result;
}

