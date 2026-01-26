/**
 * Isocontours benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "isocontours-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_isocontours_tf_benchmark(const std::vector<std::string> &mesh_paths,
                                 int n_samples, std::ostream &out) {
  out << "polygons,n_cuts,time_ms\n";

  constexpr int n_cuts_list[] = {1, 2, 4, 8, 16, 32, 64, 128};

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();

    // Create scalar field: distance from origin
    std::vector<float> scalar_field(mesh.points().size());
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < mesh.points().size(); ++i) {
      auto pt = mesh.points()[i];
      float val = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]);
      scalar_field[i] = val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    for (int n_cuts : n_cuts_list) {
      // Generate evenly spaced cut values
      std::vector<float> cut_values(n_cuts);
      for (int i = 0; i < n_cuts; ++i) {
        cut_values[i] = min_val + (max_val - min_val) * (i + 1) / (n_cuts + 1);
      }

      auto time = benchmark::min_time_of(
          [&]() {
            auto contours =
                tf::make_isocontours(polygons, tf::make_range(scalar_field),
                                     tf::make_range(cut_values));
            benchmark::do_not_optimize(contours);
          },
          n_samples);

      out << polygons.size() << "," << n_cuts << "," << time << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
