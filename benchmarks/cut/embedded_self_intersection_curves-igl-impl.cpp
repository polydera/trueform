/**
 * Embedded self-intersection curves benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "embedded_self_intersection_curves-igl.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/copyleft/cgal/remesh_self_intersections.h>

namespace benchmark {

int run_embedded_self_intersection_curves_igl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(r_polygons.polygons());
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);

    // Transform TrueForm mesh once
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());
    tf::parallel_transform(points, transformed.points(), [&](auto pt) {
      return tf::transformed(pt, transform);
    });

    // Concatenate the two meshes once
    auto concatenated_mesh =
        tf::concatenated(r_polygons.polygons(), transformed.polygons());

    // Convert to libigl format once
    auto V = benchmark::igl::to_igl_vertices(concatenated_mesh.points());
    auto F = benchmark::igl::to_igl_faces(concatenated_mesh.faces());

    auto time_ms = benchmark::min_time_of(
        [&]() {
          Eigen::MatrixXd VV;
          Eigen::MatrixXi FF;
          Eigen::MatrixXi IF;
          Eigen::VectorXi J;
          Eigen::VectorXi IM;
          ::igl::copyleft::cgal::remesh_self_intersections(V, F, {}, VV, FF, IF,
                                                           J, IM);
          benchmark::do_not_optimize(VV);
          benchmark::do_not_optimize(FF);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
