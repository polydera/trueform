/**
 * Benchmark: Embedded self-intersection curves with TrueForm
 *
 * Measures time to compute self-intersection curves on a mesh made by
 * concatenating two overlapping copies of the same mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>

int main() {
    std::cout << "polygons0,polygons1,time_ms\n";

    constexpr int n_samples = 10;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto mesh = tf::read_stl<int>(path);
        auto polygons = mesh.polygons();
        auto points = mesh.points();

        tf::polygons_buffer<int, float, 3, 3> transformed;
        tf::polygons_buffer<int, float, 3, 3> concatenated_mesh;

        // Topology for concatenated mesh (rebuilt each sample)
        tf::face_membership<int> fm;
        tf::manifold_edge_link<int, 3> mel;
        tf::tree<int, float, 3> tree;

        auto time_ms = benchmark::mean_time_of(
            [&]() {
                // Pick a random "pivot" point for the random transformation
                auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto pivot = points[pivot_idx];
                auto new_origin_id = tf::random<int>(0, static_cast<int>(points.size()));
                auto new_origin = points[new_origin_id];

                auto transform = tf::random_transformation_at(pivot, new_origin);

                // Transform TrueForm mesh
                transformed.faces_buffer() = mesh.faces_buffer();
                transformed.points_buffer().allocate(points.size());
                tf::parallel_transform(points, transformed.points(), [&](auto pt) {
                    return tf::transformed(pt, transform);
                });

                // Concatenate the two meshes
                concatenated_mesh = tf::concatenated(polygons, transformed.polygons());

                // Build topology on concatenated mesh
                fm.build(concatenated_mesh.polygons());
                mel.build(concatenated_mesh.polygons().faces(), fm);
                tree.build(concatenated_mesh.polygons(), tf::config_tree(4, 4));
            },
            [&]() {
                auto result = tf::embedded_self_intersection_curves(
                    concatenated_mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel));
                benchmark::do_not_optimize(result);
            },
            n_samples);

        std::cout << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
    }

    return 0;
}
