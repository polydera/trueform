/**
 * Embedded self-intersection curves benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "embedded_self_intersection_curves-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_embedded_self_intersection_curves_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    auto points = mesh.points();

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(polygons);
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);

    // Transform TrueForm mesh once
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = mesh.faces_buffer();
    transformed.points_buffer().allocate(points.size());
    tf::parallel_transform(points, transformed.points(), [&](auto pt) {
      return tf::transformed(pt, transform);
    });

    // Concatenate the two meshes once
    auto concatenated_mesh =
        tf::concatenated(polygons, transformed.polygons());

    // Build topology on concatenated mesh once
    tf::face_membership<int> fm;
    fm.build(concatenated_mesh.polygons());
    tf::manifold_edge_link<int, 3> mel;
    mel.build(concatenated_mesh.polygons().faces(), fm);
    tf::tree<int, float, 3> tree;
    tree.build(concatenated_mesh.polygons(), tf::config_tree(4, 4));

    auto time_ms = benchmark::min_time_of(
        [&]() {
          auto result = tf::embedded_self_intersection_curves(
              concatenated_mesh.polygons() | tf::tag(tree) | tf::tag(fm) |
              tf::tag(mel));
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
