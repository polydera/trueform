/**
 * Benchmark: Boolean operations with CGAL
 *
 * Measures time to compute boolean union between two meshes
 * using CGAL's PMP::corefine_and_compute_union.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <CGAL/Polygon_mesh_processing/corefinement.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

int main() {
    std::cout << "polygons0,polygons1,time_ms\n";

    constexpr int n_samples = 10;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);
        auto points = r_polygons.points();

        // Create mesh1
        auto mesh1 = benchmark::cgal::to_cgal_mesh_d(r_polygons);

        Surface_mesh_d mesh2;
        auto time_ms = benchmark::mean_time_of(
            [&]() {
                // Pick a random "pivot" point for the random transformation
                auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto pivot = points[pivot_idx];
                auto new_origin_id = tf::random<int>(0, static_cast<int>(points.size()));
                auto new_origin = points[new_origin_id];

                auto transform = tf::random_transformation_at(pivot, new_origin);

                // Transform TrueForm mesh, then convert to CGAL
                tf::polygons_buffer<int, float, 3, 3> transformed;
                transformed.faces_buffer() = r_polygons.faces_buffer();
                transformed.points_buffer().allocate(points.size());
                tf::parallel_transform(points, transformed.points(), [&](auto pt) {
                    return tf::transformed(pt, transform);
                });

                mesh2 = benchmark::cgal::to_cgal_mesh_d(transformed);
            },
            [&]() {
                Surface_mesh_d mesh1_copy = mesh1;
                Surface_mesh_d mesh2_copy = mesh2;
                Surface_mesh result;
                PMP::corefine_and_compute_union(mesh1_copy, mesh2_copy, result);
                benchmark::do_not_optimize(result);
            },
            n_samples);

        std::cout << r_polygons.faces().size() << "," << r_polygons.faces().size() << "," << time_ms << "\n";
    }

    return 0;
}
