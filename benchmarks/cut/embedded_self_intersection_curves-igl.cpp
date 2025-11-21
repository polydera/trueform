/**
 * Benchmark: Embedded self-intersection curves with libigl
 *
 * Measures time to compute self-intersection resolution on a mesh made by
 * concatenating two overlapping copies using igl::copyleft::cgal::remesh_self_intersections.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <igl/copyleft/cgal/remesh_self_intersections.h>

int main() {
    std::cout << "polygons0,polygons1,time_ms\n";

    constexpr int n_samples = 10;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);
        auto points = r_polygons.points();

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;

        auto time_ms = benchmark::mean_time_of(
            [&]() {
                // Pick a random "pivot" point for the random transformation
                auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto pivot = points[pivot_idx];
                auto new_origin_id = tf::random<int>(0, static_cast<int>(points.size()));
                auto new_origin = points[new_origin_id];

                auto transform = tf::random_transformation_at(pivot, new_origin);

                // Transform TrueForm mesh
                tf::polygons_buffer<int, float, 3, 3> transformed;
                transformed.faces_buffer() = r_polygons.faces_buffer();
                transformed.points_buffer().allocate(points.size());
                tf::parallel_transform(points, transformed.points(), [&](auto pt) {
                    return tf::transformed(pt, transform);
                });

                // Concatenate the two meshes
                auto concatenated_mesh = tf::concatenated(r_polygons.polygons(), transformed.polygons());

                // Convert to libigl format
                V = benchmark::igl::to_igl_vertices(concatenated_mesh.points());
                F = benchmark::igl::to_igl_faces(concatenated_mesh.faces());
            },
            [&]() {
                Eigen::MatrixXd VV;
                Eigen::MatrixXi FF;
                Eigen::MatrixXi IF;
                Eigen::VectorXi J;
                Eigen::VectorXi IM;
                igl::copyleft::cgal::remesh_self_intersections(V, F, {}, VV, FF, IF, J, IM);
                benchmark::do_not_optimize(VV);
                benchmark::do_not_optimize(FF);
            },
            n_samples);

        std::cout << r_polygons.faces().size() << "," << r_polygons.faces().size() << "," << time_ms << "\n";
    }

    return 0;
}
