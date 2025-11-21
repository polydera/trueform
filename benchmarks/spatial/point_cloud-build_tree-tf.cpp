/**
 * Benchmark: Point cloud tree building with TrueForm
 *
 * Measures time to build spatial acceleration structure (KD-tree)
 * on point clouds of varying sizes using TrueForm's tf::tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>

int main() {
    std::cout << "points,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto polygons = tf::read_stl<int>(path);
        auto points = polygons.points();

        auto time = benchmark::min_time_of([&]() {
            tf::tree<int, float, 3> tree;
            tree.build(points, tf::config_tree(4, 4));
            benchmark::do_not_optimize(tree);
        });

        std::cout << points.size() << "," << time << "\n";
    }

    return 0;
}
