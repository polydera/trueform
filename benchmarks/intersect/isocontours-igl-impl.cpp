/**
 * Isocontours benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "isocontours-igl.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/isolines.h>

namespace benchmark {

int run_isocontours_igl_benchmark(const std::vector<std::string> &mesh_paths,
                                  int n_samples, std::ostream &out) {
  out << "polygons,n_cuts,time_ms\n";

  constexpr int n_cuts_list[] = {1, 2, 4, 8, 16, 32, 64, 128};

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    // Convert to libigl format
    auto V = benchmark::igl::to_igl_vertices(r_polygons.points());
    auto F = benchmark::igl::to_igl_faces(r_polygons.faces());

    // Create scalar field: distance from origin
    Eigen::VectorXd S(V.rows());
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    for (int i = 0; i < V.rows(); ++i) {
      double val = V.row(i).norm();
      S(i) = val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    for (int n_cuts : n_cuts_list) {
      // Generate evenly spaced cut values
      Eigen::VectorXd vals(n_cuts);
      for (int i = 0; i < n_cuts; ++i) {
        vals(i) = min_val + (max_val - min_val) * (i + 1) / (n_cuts + 1);
      }

      auto time = benchmark::min_time_of(
          [&]() {
            Eigen::MatrixXd iV;
            Eigen::MatrixXi iE;
            Eigen::VectorXi I;
            ::igl::isolines(V, F, S, vals, iV, iE, I);
            benchmark::do_not_optimize(iV);
            benchmark::do_not_optimize(iE);
          },
          n_samples);

      out << r_polygons.faces().size() << "," << n_cuts << "," << time << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
