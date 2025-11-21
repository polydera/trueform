/**
 * Benchmark: Point cloud kNN queries with TrueForm
 *
 * Measures time to perform k-nearest neighbor queries on point clouds
 * of varying sizes using TrueForm's tf::tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>

int main() {
    std::cout << "points,k,time_ms\n";

    constexpr int max_k = 10;
    constexpr int n_samples = 100;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto polygons = tf::read_stl<int>(path);
        auto points = polygons.points();

        // Build tree
        tf::tree<int, float, 3> tree;
        tree.build(points, tf::config_tree(4, 4));

        // Calculate diagonal length for query generation
        auto l = tf::aabb_from(points).diagonal().length();

        // Query point (updated in prepare)
        tf::point<float, 3> query_point;
        auto form = tf::make_form(tree, points);

        // Buffer for kNN results
        std::array<tf::nearest_neighbor<int, float, 3>, max_k> buffer;

        for (int k = 1; k <= max_k; ++k) {
            auto time = benchmark::mean_time_of(
                [&]() {
                    // Prepare: generate random query point
                    auto idx = tf::random<int>(0, points.size());
                    auto point = points[idx];
                    query_point = point + tf::random_vector<float, 3>() * l;
                },
                [&]() {
                    // Benchmark: kNN query
                    auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
                    tf::neighbor_search(form, query_point, knn);
                    benchmark::do_not_optimize(knn);
                }, n_samples
            );

            std::cout << points.size() << "," << k << "," << time << "\n";
        }
    }

    return 0;
}
