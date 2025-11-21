/**
 * Benchmark: Boolean operations with libigl
 *
 * Measures time to compute boolean union between two meshes
 * using libigl's mesh_boolean.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <igl/copyleft/cgal/mesh_boolean.h>

int main() {
    std::cout << "polygons0,polygons1,time_ms\n";

    constexpr int n_samples = 10;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);
        auto points = r_polygons.points();

        // Convert to libigl format
        auto V1 = benchmark::igl::to_igl_vertices(r_polygons.points());
        auto F1 = benchmark::igl::to_igl_faces(r_polygons.faces());

        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;
        auto time_ms = benchmark::mean_time_of(
            [&]() {
                // Pick a random "pivot" point for the random transformation
                auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto pivot = points[pivot_idx];
                auto new_origin_id = tf::random<int>(0, static_cast<int>(points.size()));
                auto new_origin = points[new_origin_id];

                auto transform = tf::random_transformation_at(pivot, new_origin);

                // Transform TrueForm mesh, then convert to libigl
                tf::polygons_buffer<int, float, 3, 3> transformed;
                transformed.faces_buffer() = r_polygons.faces_buffer();
                transformed.points_buffer().allocate(points.size());
                tf::parallel_transform(points, transformed.points(), [&](auto pt) {
                    return tf::transformed(pt, transform);
                });

                V2 = benchmark::igl::to_igl_vertices(transformed.points());
                F2 = benchmark::igl::to_igl_faces(transformed.faces());
            },
            [&]() {
                Eigen::MatrixXd VC;
                Eigen::MatrixXi FC;
                igl::copyleft::cgal::mesh_boolean(V1, F1, V2, F2, igl::MESH_BOOLEAN_TYPE_UNION, VC, FC);
                benchmark::do_not_optimize(VC);
                benchmark::do_not_optimize(FC);
            },
            n_samples);

        std::cout << r_polygons.faces().size() << "," << r_polygons.faces().size() << "," << time_ms << "\n";
    }

    return 0;
}
