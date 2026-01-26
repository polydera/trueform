/**
 * Polygons to polygons closest-point Coal - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons_to_polygons-closest_point-coal.hpp"
#include "timing.hpp"
#include <memory>
#include <trueform/trueform.hpp>

using namespace benchmark::coal;

namespace benchmark {

int run_polygons_to_polygons_closest_point_coal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {

  out << "bv,polygons,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);

    auto [coal_vertices, coal_triangles] =
        benchmark::coal::to_coal_geometry(polygons);

    if (coal_vertices.empty() || coal_triangles.empty()) {
      continue;
    }

    auto points = polygons.points();
    auto n_pts = points.size();
    auto n_faces = coal_triangles.size();
    auto n_tris = static_cast<int>(coal_triangles.size());
    auto n_verts = static_cast<int>(coal_vertices.size());

    DistanceRequest dreq;
    dreq.enable_nearest_points = true;
    DistanceResult dres;
    auto l = tf::aabb_from(points).diagonal().length();
    Transform3 tf_moving = Transform3::Identity();

    // auto model_aabb = std::make_shared<Model_AABB>();
    // model_aabb->beginModel(n_tris, n_verts);
    // model_aabb->addSubModel(coal_vertices, coal_triangles);
    // model_aabb->endModel();
    //
    // ::coal::CollisionObject obj_fixed_aabb(model_aabb);
    // ::coal::CollisionObject obj_moving_aabb(model_aabb);
    // obj_fixed_aabb.setTransform(Transform3::Identity());
    //
    // auto time_aabb = benchmark::mean_time_of(
    //     [&]() {
    //       auto pivot_idx = tf::random<int>(0, static_cast<int>(n_pts) - 1);
    //       auto pivot = points[pivot_idx];
    //       auto translation = pivot + tf::random_vector<float, 3>() * 2 * l;
    //       auto T = tf::random_transformation_at(pivot, translation);
    //
    //       tf_moving.setIdentity();
    //       for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //           tf_moving.rotation()(i, j) = static_cast<Scalar>(T(i, j));
    //         }
    //       }
    //       tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
    //           static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));
    //       obj_moving_aabb.setTransform(tf_moving);
    //     },
    //     [&]() {
    //       dres.clear();
    //       Scalar dist =
    //           ::coal::distance(&obj_fixed_aabb, &obj_moving_aabb, dreq, dres);
    //       benchmark::do_not_optimize(dist);
    //       benchmark::do_not_optimize(dres.min_distance);
    //     },
    //     n_samples);
    // out << "AABB," << n_faces << "," << n_faces << "," << time_aabb << "\n";

    auto model_obbrss = std::make_shared<Model_OBBRSS>();
    model_obbrss->beginModel(n_tris, n_verts);
    model_obbrss->addSubModel(coal_vertices, coal_triangles);
    model_obbrss->endModel();

    ::coal::CollisionObject obj_fixed_obbrss(model_obbrss);
    ::coal::CollisionObject obj_moving_obbrss(model_obbrss);
    obj_fixed_obbrss.setTransform(Transform3::Identity());

    auto time_obbrss = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(n_pts) - 1);
          auto pivot = points[pivot_idx];
          auto translation = pivot + tf::random_vector<float, 3>() * 2 * l;
          auto T = tf::random_transformation_at(pivot, translation);

          tf_moving.setIdentity();
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              tf_moving.rotation()(i, j) = static_cast<Scalar>(T(i, j));
            }
          }
          tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
              static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));
          obj_moving_obbrss.setTransform(tf_moving);
        },
        [&]() {
          dres.clear();
          Scalar dist = ::coal::distance(&obj_fixed_obbrss, &obj_moving_obbrss,
                                        dreq, dres);
          benchmark::do_not_optimize(dist);
          benchmark::do_not_optimize(dres.min_distance);
        },
        n_samples);
    out << "OBBRSS," << n_faces << "," << n_faces << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
