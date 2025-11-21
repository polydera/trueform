/**
 * Benchmark: Mesh–mesh distance (FCL)
 *
 * - Start from a triangle mesh (TrueForm polygons)
 * - Copy points + triangle indices into an FCL BVH model
 * - Build the BVH once
 * - Two instances of the same mesh: fixed and moving
 * - Each sample:
 *     - choose a random pivot on the mesh
 *     - build a random rigid transform at that pivot (TrueForm)
 *     - convert it to an FCL/Eigen transform for the moving mesh
 *     - run one mesh–mesh distance query with FCL
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>
#include <memory>

// Use FCL type aliases from conversions.hpp
using namespace benchmark::fcl;

int main() {
  std::cout << "polygons,polygons,time_ms\n";

  constexpr int n_samples = 1;

  for (const auto &path : benchmark::BENCHMARK_MESHES) {
    // Load mesh as TrueForm polygons
    auto polygons = tf::read_stl<int>(path);

    // Convert to FCL geometry
    auto [fcl_vertices, fcl_triangles] =
        benchmark::fcl::to_fcl_geometry(polygons);

    if (fcl_vertices.empty() || fcl_triangles.empty()) {
      continue; // skipped due to empty or non-triangular mesh
    }

    auto points = polygons.points();
    auto n_pts = points.size();
    auto n_faces = fcl_triangles.size();

    // Build FCL BVH model once
    auto model = std::make_shared<Model>();
    model->beginModel(static_cast<int>(fcl_triangles.size()),
                      static_cast<int>(fcl_vertices.size()));
    model->addSubModel(fcl_vertices, fcl_triangles);
    model->endModel();

    // Two instances of the same geometry (mesh vs. transformed mesh)
    fcl::CollisionObject<Scalar> obj_fixed(model);  // stays at identity
    fcl::CollisionObject<Scalar> obj_moving(model); // gets transformed

    Transform3 tf_fixed = Transform3::Identity();
    Transform3 tf_moving = Transform3::Identity();

    obj_fixed.setTransform(tf_fixed);

    // Distance request/result reused across samples
    DistanceRequest dreq;
    dreq.enable_nearest_points = true;
    DistanceResult dres;
    auto l = tf::aabb_from(points).diagonal().length();



 // Build tree
    tf::tree<int, float, 3> tree;
    tree.build(polygons.polygons(), tf::config_tree(4, 4));

    auto form_polygons = tf::make_form(tree, polygons.polygons());
    tf::frame<float, 3> frame;


    // Benchmark: each sample picks a new random rigid transform for the moving
    // mesh
    auto time_ms = benchmark::mean_time_of(
        // prepare: choose a new random pivot + rigid transform at that pivot
        [&]() {
          // Random pivot vertex
          auto pivot_idx = tf::random<int>(0, static_cast<int>(n_pts));
          auto pivot = points[pivot_idx];

          auto translation = pivot + tf::random_vector<float, 3>() * 2 * l;
          // TrueForm 4x4 rigid transform at pivot
          auto T = tf::random_transformation_at(pivot, translation);

          // Map TF transform to FCL/Eigen Transform3:
          // T(i,j) for i,j = 0..2 => rotation, T(3,j) => translation
          tf_moving.setIdentity();
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              tf_moving.linear()(i, j) = static_cast<Scalar>(T(i, j));
            }
          }
          tf_moving.translation() << static_cast<Scalar>(T(0, 3)),
              static_cast<Scalar>(T(1, 3)), static_cast<Scalar>(T(2, 3));

          obj_moving.setTransform(tf_moving);
        },
        // body: single FCL mesh–mesh distance query
        [&]() {
          dres.clear();
          Scalar dist = fcl::distance(&obj_fixed, &obj_moving, dreq, dres);

          benchmark::do_not_optimize(dist);
          benchmark::do_not_optimize(dres.min_distance);
          if (dreq.enable_nearest_points) {
            benchmark::do_not_optimize(dres.nearest_points[0]);
            benchmark::do_not_optimize(dres.nearest_points[1]);
          }
        },
        n_samples);

    std::cout << n_faces << "," << n_faces << "," << time_ms << "\n";
  }

  return 0;
}
