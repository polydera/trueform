/**
 * Boolean operations benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-tf.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_boolean_tf_benchmark(const std::vector<std::string> &mesh_paths,
                             int n_samples, std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    auto points = mesh.points();

    tf::face_membership<int> fm;
    fm.build(polygons);
    tf::manifold_edge_link<int, 3> mel;
    mel.build(polygons.faces(), fm);
    tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));

    auto tagged = polygons | tf::tag(mel) | tf::tag(fm);
    auto form0 = tagged | tf::tag(tree);

    // Rotation around centroid, using smallest axis
    auto aabb = tf::aabb_from(polygons);
    auto pivot = tf::centroid(polygons);
    auto diag = aabb.diagonal();
    auto inv_diag =
        tf::vector<float, 3>{1.0f / diag[0], 1.0f / diag[1], 1.0f / diag[2]};
    int rot_axis = tf::largest_axis(inv_diag);

    tf::frame<float, 3> frame;
    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.5f) / float(n_samples)};
          frame = tf::make_frame(
              benchmark::make_rotation(angle, rot_axis, pivot));
          ++iter;
        },
        [&]() {
          auto form1 = tagged | tf::tag(tree) | tf::tag(frame);
          auto result = tf::make_boolean(form0, form1, tf::boolean_op::merge);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
