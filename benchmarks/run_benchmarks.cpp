/**
 * TrueForm Benchmarks - Unified Runner
 *
 * Runs all benchmark suites in sequence, outputting results to stdout
 * or separate CSV files.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Include all benchmark headers
// Cut module
#include "cut/boolean-cgal.hpp"
#include "cut/boolean-igl.hpp"
#include "cut/boolean-tf.hpp"
#include "cut/embedded_isocurves-tf.hpp"
#include "cut/embedded_self_intersection_curves-igl.hpp"
#include "cut/embedded_self_intersection_curves-tf.hpp"
#ifdef HAVE_VTK
#include "cut/embedded_isocurves-vtk.hpp"
#endif

// Topology module
#include "topology/boundary_paths-cgal.hpp"
#include "topology/boundary_paths-igl.hpp"
#include "topology/boundary_paths-tf.hpp"
#include "topology/connected_components-cgal.hpp"
#include "topology/connected_components-igl.hpp"
#include "topology/connected_components-tf.hpp"
#ifdef HAVE_VTK
#include "topology/boundary_paths-vtk.hpp"
#include "topology/connected_components-vtk.hpp"
#endif

// Intersect module
#include "intersect/isocontours-igl.hpp"
#include "intersect/isocontours-tf.hpp"
#include "intersect/mesh_mesh_curves-cgal.hpp"
#include "intersect/mesh_mesh_curves-tf.hpp"
#ifdef HAVE_VTK
#include "intersect/isocontours-vtk.hpp"
// #include "intersect/mesh_mesh_curves-vtk.hpp"
#endif

// Spatial module
#include "spatial/point_cloud-build_tree-nanoflann.hpp"
#include "spatial/point_cloud-build_tree-tf.hpp"
#include "spatial/point_cloud-knn-nanoflann.hpp"
#include "spatial/point_cloud-knn-tf.hpp"
#include "spatial/polygons-build_tree-cgal.hpp"
#include "spatial/polygons-build_tree-fcl.hpp"
#include "spatial/polygons-build_tree-tf.hpp"
#include "spatial/mod_tree-update-tf.hpp"
#include "spatial/polygons_to_polygons-closest_point-fcl.hpp"
#include "spatial/polygons_to_polygons-closest_point-tf.hpp"

#include "common/test_meshes.hpp"

struct benchmark_info {
  std::string name;
  std::function<int(const std::vector<std::string> &, int, std::ostream &)>
      func;
  int default_samples;
};

auto parse_benchmark_name(const std::string &benchmark_name)
    -> std::pair<std::string, std::string> {
  size_t first_dash = benchmark_name.find('-');
  if (first_dash == std::string::npos) {
    return {"unknown", benchmark_name};
  }
  return {benchmark_name.substr(0, first_dash),
          benchmark_name.substr(first_dash + 1)};
}

auto create_directory_if_needed(const std::filesystem::path &path) -> bool {
  try {
    if (std::filesystem::exists(path)) {
      if (!std::filesystem::is_directory(path)) {
        std::cerr << "ERROR: Path exists but is not a directory: " << path
                  << "\n";
        return false;
      }
      return true;
    }

    if (std::filesystem::create_directories(path)) {
      std::cout << "Created directory: " << path << "\n";
      return true;
    }
    return false;
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "ERROR: Failed to create directory " << path << ": "
              << e.what() << "\n";
    return false;
  }
}

auto open_benchmark_output_file(const std::string &benchmark_name)
    -> std::ofstream {
  auto [module, name] = parse_benchmark_name(benchmark_name);

  std::filesystem::path results_dir = "results";
  if (!create_directory_if_needed(results_dir)) {
    return {};
  }

  std::filesystem::path module_dir = results_dir / module;
  if (!create_directory_if_needed(module_dir)) {
    return {};
  }

  std::filesystem::path output_file = module_dir / (name + ".csv");
  std::ofstream file(output_file);

  if (!file.is_open()) {
    std::cerr << "ERROR: Failed to open output file: " << output_file << "\n";
  } else {
    std::cout << "Writing to: " << output_file << "\n";
  }

  return file;
}

int main(int argc, char *argv[]) {
  // Register all benchmarks
  std::vector<benchmark_info> benchmarks = {
      // Cut module (7 benchmarks)
      {"cut-boolean-tf", benchmark::run_boolean_tf_benchmark, 10},
      {"cut-boolean-cgal", benchmark::run_boolean_cgal_benchmark, 10},
      {"cut-boolean-igl", benchmark::run_boolean_igl_benchmark, 10},
      {"cut-embedded_self_intersection_curves-tf",
       benchmark::run_embedded_self_intersection_curves_tf_benchmark, 10},
      {"cut-embedded_self_intersection_curves-igl",
       benchmark::run_embedded_self_intersection_curves_igl_benchmark, 10},
      {"cut-embedded_isocurves-tf",
       benchmark::run_embedded_isocurves_tf_benchmark, 10},
#ifdef HAVE_VTK
      {"cut-embedded_isocurves-vtk",
       benchmark::run_embedded_isocurves_vtk_benchmark, 10},
#endif

      // Topology module (8 benchmarks)
      {"topology-connected_components-cgal",
       benchmark::run_connected_components_cgal_benchmark, 10},
      {"topology-connected_components-tf",
       benchmark::run_connected_components_tf_benchmark, 100},
      {"topology-connected_components-igl",
       benchmark::run_connected_components_igl_benchmark, 10},
      {"topology-boundary_paths-cgal",
       benchmark::run_boundary_paths_cgal_benchmark, 10},
      {"topology-boundary_paths-tf", benchmark::run_boundary_paths_tf_benchmark,
       100},
      {"topology-boundary_paths-igl",
       benchmark::run_boundary_paths_igl_benchmark, 10},
#ifdef HAVE_VTK
      {"topology-connected_components-vtk",
       benchmark::run_connected_components_vtk_benchmark, 10},
      {"topology-boundary_paths-vtk",
       benchmark::run_boundary_paths_vtk_benchmark, 10},
#endif

      // Intersect module (6 benchmarks)
      {"intersect-mesh_mesh_curves-tf",
       benchmark::run_mesh_mesh_curves_tf_benchmark, 10},
      {"intersect-mesh_mesh_curves-cgal",
       benchmark::run_mesh_mesh_curves_cgal_benchmark, 10},
      // too slow
/*#ifdef HAVE_VTK*/
/*      {"intersect-mesh_mesh_curves-vtk",*/
/*       benchmark::run_mesh_mesh_curves_vtk_benchmark, 10},*/
/*#endif*/
      {"intersect-isocontours-tf", benchmark::run_isocontours_tf_benchmark, 10},
      {"intersect-isocontours-igl", benchmark::run_isocontours_igl_benchmark,
       10},
#ifdef HAVE_VTK
      {"intersect-isocontours-vtk", benchmark::run_isocontours_vtk_benchmark,
       10},
#endif
      // Spatial module (11 benchmarks)
      {"spatial-point_cloud-build_tree-tf",
       benchmark::run_point_cloud_build_tree_tf_benchmark, 10},
      {"spatial-point_cloud-build_tree-nanoflann",
       benchmark::run_point_cloud_build_tree_nanoflann_benchmark, 10},
      {"spatial-point_cloud-knn-tf",
       benchmark::run_point_cloud_knn_tf_benchmark, 1000},
      {"spatial-point_cloud-knn-nanoflann",
       benchmark::run_point_cloud_knn_nanoflann_benchmark, 1000},
      {"spatial-polygons-build_tree-tf",
       benchmark::run_polygons_build_tree_tf_benchmark, 10},
      {"spatial-polygons-build_tree-cgal",
       benchmark::run_polygons_build_tree_cgal_benchmark, 10},
      {"spatial-polygons-build_tree-fcl",
       benchmark::run_polygons_build_tree_fcl_benchmark, 10},
      {"spatial-polygons_to_polygons-closest_point-tf",
       benchmark::run_polygons_to_polygons_closest_point_tf_benchmark, 1000},
      {"spatial-polygons_to_polygons-closest_point-fcl",
       benchmark::run_polygons_to_polygons_closest_point_fcl_benchmark, 1000},
      {"spatial-mod_tree-update-tf",
       benchmark::run_mod_tree_update_tf_benchmark, 10},
  };

  // Simple CLI: run all benchmarks with default mesh paths and samples
  std::cout << "TrueForm Benchmarks - Unified Runner\n";
  std::cout << "=====================================\n\n";

  int total_benchmarks = benchmarks.size();
  int completed = 0;

  for (const auto &bench : benchmarks) {
    std::cout << "Running [" << (completed + 1) << "/" << total_benchmarks
              << "]: " << bench.name << " (n_samples=" << bench.default_samples
              << ")\n";

    auto output_file = open_benchmark_output_file(bench.name);
    if (!output_file.is_open()) {
      std::cerr << "ERROR: Failed to create output file for " << bench.name
                << "\n";
      return 1;
    }

    output_file << std::unitbuf; // Disable buffering for real-time progress

    int result = bench.func(benchmark::BENCHMARK_MESHES, bench.default_samples,
                            output_file);

    if (result != 0) {
      std::cerr << "ERROR: Benchmark " << bench.name << " failed with code "
                << result << "\n";
      return 1;
    }

    std::cout << "Completed successfully\n\n";
    completed++;
  }

  std::cout << "=====================================\n";
  std::cout << "All " << total_benchmarks
            << " benchmarks completed successfully!\n";

  return 0;
}
