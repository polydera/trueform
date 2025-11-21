/**
 * Benchmark: Closest point pair between point cloud and mesh (CGAL)
 *
 * - Start from a triangle mesh (TrueForm polygons)
 * - Each sample:
 *    - copy mesh points to a point cloud
 *    - apply random transform + translation
 * - Build a CGAL AABB tree on the original mesh (once)
 * - Find the closest pair (point in cloud, closest point on mesh)
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "test_meshes.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <iostream>
#include <limits>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

using namespace benchmark::cgal;

// CGAL type aliases
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits    = CGAL::AABB_traits<Kernel, Primitive>;
using Tree      = CGAL::AABB_tree<Traits>;
using Point_3   = Kernel::Point_3;

struct ClosestPairResult {
  double      sqdist      = std::numeric_limits<double>::infinity();
  Point_3     cloud_point;
  Point_3     mesh_point;
  std::size_t cloud_index = 0;
};

int main() {
  std::cout << "polygons,points,time_ms\n";

  constexpr int n_samples = 10;

  for (const auto &path : benchmark::BENCHMARK_MESHES) {
    // Load mesh as TrueForm polygons
    auto polygons = tf::read_stl<int>(path);
    auto points   = polygons.points();  // range view
    auto n_pts    = points.size();


    // We'll reuse this buffer across samples; contents rebuilt in prepare()
    auto point_cloud = polygons.points_buffer();  // initial shape; values overwritten

    // Precompute scale length for translation
    auto l = tf::aabb_from(points).diagonal().length();

    // Convert mesh to CGAL + build AABB tree on *original* mesh (once)
    auto mesh = benchmark::cgal::to_cgal_mesh(polygons);

    Tree tree(faces(mesh).begin(), faces(mesh).end(), mesh);
    tree.build();
    tree.accelerate_distance_queries(); // optimize distance queries

    // Benchmark: each sample builds a new transformed cloud and scans it
    auto time_ms = benchmark::mean_time_of(
        // prepare: rebuild point cloud with fresh random transform
        [&]() {
          // 1) copy original points into point_cloud
          //    (assumes same size / layout as polygons.points_buffer())
          std::size_t i = 0;
          for (const auto &p_src : points) {
            point_cloud[i++] = p_src;
          }

          // 2) random pivot and transform
          auto pivot_idx     = tf::random<int>(0, static_cast<int>(n_pts));
          auto pivot         = points[pivot_idx];
          auto transformation = tf::random_transformation_at(pivot);

          // 3) random translation (vector), scaled by bbox diagonal
          auto translation = tf::random_vector<float, 3>() * l;

          // 4) apply transform + translation to all copied points
          //    p' = transformed(p, T) + translation
          for (auto &&p : point_cloud) {
            p = tf::transformed(p, transformation) + translation;
          }
        },
        // body: scan entire point cloud once, track global min distance
        [&]() {
          ClosestPairResult best;

          std::size_t idx = 0;
          for (const auto &p_tf : point_cloud) {
            // Convert TF point to CGAL point
            Point_3 q(p_tf[0], p_tf[1], p_tf[2]);

            // Closest point on mesh
            Point_3 cp = tree.closest_point(q);

            double d2 = CGAL::squared_distance(q, cp);
            if (d2 < best.sqdist) {
              best.sqdist     = d2;
              best.cloud_point = q;
              best.mesh_point  = cp;
              best.cloud_index = idx;
            }
            ++idx;
          }

          benchmark::do_not_optimize(best.sqdist);
          benchmark::do_not_optimize(best.cloud_point);
          benchmark::do_not_optimize(best.mesh_point);
        },
        n_samples);

    std::cout << polygons.faces().size() << "," << n_pts << "," << time_ms
              << "\n";
  }

  return 0;
}

