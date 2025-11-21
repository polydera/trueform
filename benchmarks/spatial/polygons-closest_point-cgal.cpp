/**
 * Benchmark: Closest-point queries with CGAL AABB tree
 *
 * Measures time to compute closest points from random queries to a triangle mesh
 * using CGAL's AABB tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"

#include <iostream>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

// CGAL Type Definitions
using namespace benchmark::cgal;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits    = CGAL::AABB_traits<Kernel, Primitive>;
using Tree      = CGAL::AABB_tree<Traits>;
using Point_3   = Kernel::Point_3;

int main() {
    std::cout << "polygons,time_ms\n";

    constexpr int n_samples = 100;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        // Load TrueForm polygons
        auto polygons = tf::read_stl<int>(path);
        auto points  = polygons.points();

        // Convert to CGAL mesh
        auto mesh = benchmark::cgal::to_cgal_mesh(polygons);

        // Build AABB tree once per mesh
        Tree tree(faces(mesh).begin(), faces(mesh).end(), mesh);
        tree.build();
        tree.accelerate_distance_queries(); // internal structure for distance queries

        // Bbox diagonal length for offset scale (same logic as nanoflann benchmark)
        auto l = tf::aabb_from(polygons.points()).diagonal().length();

        // Query point (updated in prepare lambda)
        Point_3 query_point;

        auto time = benchmark::mean_time_of(
            // prepare: generate random query point
            [&]() {
                auto idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto p   = points[idx];

                auto random_offset = tf::random_vector<float, 3>() * l;

                query_point = Point_3(
                    p[0] + random_offset[0],
                    p[1] + random_offset[1],
                    p[2] + random_offset[2]
                );
            },
            // measured: closest-point query
            [&]() {
                auto closest = tree.closest_point(query_point);
                benchmark::do_not_optimize(closest);
            },
            n_samples
        );

        std::cout << polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}

