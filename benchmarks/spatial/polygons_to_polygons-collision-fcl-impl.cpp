/**
 * Polygons to polygons collision FCL - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons_to_polygons-collision-fcl.hpp"
#include "timing.hpp"
#include <memory>
#include <trueform/trueform.hpp>

using namespace benchmark::fcl;

namespace benchmark {

int run_polygons_to_polygons_collision_fcl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {

  out << "bv,polygons,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);

    auto [fcl_vertices, fcl_triangles] =
        benchmark::fcl::to_fcl_geometry(polygons);

    if (fcl_vertices.empty() || fcl_triangles.empty()) {
      continue;
    }

    auto points = polygons.points();
    auto n_pts = points.size();
    auto n_faces = fcl_triangles.size();
    auto n_tris = static_cast<int>(fcl_triangles.size());
    auto n_verts = static_cast<int>(fcl_vertices.size());

    CollisionRequest creq;
    CollisionResult cres;
    auto l = tf::aabb_from(points).diagonal().length();
    Transform3 tf_moving = Transform3::Identity();

    auto model_aabb = std::make_shared<Model_AABB>();
    model_aabb->beginModel(n_tris, n_verts);
    model_aabb->addSubModel(fcl_vertices, fcl_triangles);
    model_aabb->endModel();

    ::fcl::CollisionObject<Scalar> obj_fixed_aabb(model_aabb);
    ::fcl::CollisionObject<Scalar> obj_moving_aabb(model_aabb);
    obj_fixed_aabb.setTransform(Transform3::Identity());

    auto time_aabb = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(n_pts) - 1);
          auto pivot = points[pivot_idx];
          auto translation = pivot + tf::random_vector<float, 3>() * 2 * l;
          auto T = tf::random_transformation_at(pivot, translation);

          tf_moving.setIdentity();
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              tf_moving.linear()(i, j) = static_cast<Scalar>(T(i, j));
            }
          }
          tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
              static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));
          obj_moving_aabb.setTransform(tf_moving);
        },
        [&]() {
          cres.clear();
          std::size_t num_contacts =
              ::fcl::collide(&obj_fixed_aabb, &obj_moving_aabb, creq, cres);
          benchmark::do_not_optimize(num_contacts);
          benchmark::do_not_optimize(cres.isCollision());
        },
        n_samples);
    out << "AABB," << n_faces << "," << n_faces << "," << time_aabb << "\n";

    auto model_obb = std::make_shared<Model_OBB>();
    model_obb->beginModel(n_tris, n_verts);
    model_obb->addSubModel(fcl_vertices, fcl_triangles);
    model_obb->endModel();

    ::fcl::CollisionObject<Scalar> obj_fixed_obb(model_obb);
    ::fcl::CollisionObject<Scalar> obj_moving_obb(model_obb);
    obj_fixed_obb.setTransform(Transform3::Identity());

    auto time_obb = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(n_pts) - 1);
          auto pivot = points[pivot_idx];
          auto translation = pivot + tf::random_vector<float, 3>() * 2 * l;
          auto T = tf::random_transformation_at(pivot, translation);

          tf_moving.setIdentity();
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              tf_moving.linear()(i, j) = static_cast<Scalar>(T(i, j));
            }
          }
          tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
              static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));
          obj_moving_obb.setTransform(tf_moving);
        },
        [&]() {
          cres.clear();
          std::size_t num_contacts =
              ::fcl::collide(&obj_fixed_obb, &obj_moving_obb, creq, cres);
          benchmark::do_not_optimize(num_contacts);
          benchmark::do_not_optimize(cres.isCollision());
        },
        n_samples);
    out << "OBB," << n_faces << "," << n_faces << "," << time_obb << "\n";

    auto model_obbrss = std::make_shared<Model_OBBRSS>();
    model_obbrss->beginModel(n_tris, n_verts);
    model_obbrss->addSubModel(fcl_vertices, fcl_triangles);
    model_obbrss->endModel();

    ::fcl::CollisionObject<Scalar> obj_fixed_obbrss(model_obbrss);
    ::fcl::CollisionObject<Scalar> obj_moving_obbrss(model_obbrss);
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
              tf_moving.linear()(i, j) = static_cast<Scalar>(T(i, j));
            }
          }
          tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
              static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));
          obj_moving_obbrss.setTransform(tf_moving);
        },
        [&]() {
          cres.clear();
          std::size_t num_contacts = ::fcl::collide(&obj_fixed_obbrss, &obj_moving_obbrss,
                                        creq, cres);
          benchmark::do_not_optimize(num_contacts);
          benchmark::do_not_optimize(cres.isCollision());
        },
        n_samples);
    out << "OBBRSS," << n_faces << "," << n_faces << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
