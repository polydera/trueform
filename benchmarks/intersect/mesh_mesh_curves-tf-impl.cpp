/**
 * Mesh-mesh intersection curves benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "mesh_mesh_curves-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_mesh_mesh_curves_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    auto points = mesh.points();

    // Build topology for mesh
    tf::face_membership<int> fm;
    fm.build(polygons);
    tf::manifold_edge_link<int, 3> mel;
    mel.build(polygons.faces(), fm);
    tf::tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
    auto form1 = tf::make_form(tree, polygons | tf::tag(mel) | tf::tag(fm));

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(polygons);
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);
    auto frame = tf::make_frame(transform);

    auto time_ms = benchmark::min_time_of(
        [&]() {
          // Compute intersection curves with transformed mesh
          auto form2 =
              tf::make_form(frame, tree, polygons | tf::tag(mel) | tf::tag(fm));
          auto curves = tf::make_intersection_curves(form1, form2);
          benchmark::do_not_optimize(curves);
        },
        n_samples);

    out << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
