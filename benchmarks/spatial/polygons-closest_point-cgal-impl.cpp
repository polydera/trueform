/**
 * Polygons closest-point CGAL - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons-closest_point-cgal.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>

using namespace benchmark::cgal;

// CGAL Type Definitions
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits = CGAL::AABB_traits_3<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;
using Point_3 = Kernel::Point_3;

namespace benchmark {

int run_polygons_closest_point_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    // Load TrueForm polygons
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    // Convert to CGAL mesh
    auto mesh = benchmark::cgal::to_cgal_mesh(polygons);

    // Build AABB tree once per mesh
    Tree tree(faces(mesh).begin(), faces(mesh).end(), mesh);
    tree.build();
    tree.accelerate_distance_queries(); // internal structure for distance
                                        // queries

    // Bbox diagonal length for offset scale (same logic as nanoflann benchmark)
    auto l = tf::aabb_from(polygons.points()).diagonal().length();

    // Query point (updated in prepare lambda)
    Point_3 query_point;

    auto time = benchmark::mean_time_of(
        // prepare: generate random query point
        [&]() {
          auto idx = tf::random<int>(0, static_cast<int>(points.size()) - 1);
          auto p = points[idx];

          auto random_offset = tf::random_vector<float, 3>() * l;

          query_point =
              Point_3(p[0] + random_offset[0], p[1] + random_offset[1],
                      p[2] + random_offset[2]);
        },
        // measured: closest-point query
        [&]() {
          auto closest = tree.closest_point(query_point);
          benchmark::do_not_optimize(closest);
        },
        n_samples);

    out << polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
