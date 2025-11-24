/**
 * Boolean operations benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-igl.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/copyleft/cgal/mesh_boolean.h>

namespace benchmark {

int run_boolean_igl_benchmark(const std::vector<std::string> &mesh_paths,
                              int n_samples, std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    // Convert to libigl format
    auto V1 = benchmark::igl::to_igl_vertices(r_polygons.points());
    auto F1 = benchmark::igl::to_igl_faces(r_polygons.faces());

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(r_polygons.polygons());
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);

    // Transform TrueForm mesh once, then convert to libigl
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());
    tf::parallel_transform(points, transformed.points(), [&](auto pt) {
      return tf::transformed(pt, transform);
    });

    auto V2 = benchmark::igl::to_igl_vertices(transformed.points());
    auto F2 = benchmark::igl::to_igl_faces(transformed.faces());

    auto time_ms = benchmark::min_time_of(
        [&]() {
          Eigen::MatrixXd VC;
          Eigen::MatrixXi FC;
          ::igl::copyleft::cgal::mesh_boolean(
              V1, F1, V2, F2, ::igl::MESH_BOOLEAN_TYPE_UNION, VC, FC);
          benchmark::do_not_optimize(VC);
          benchmark::do_not_optimize(FC);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
