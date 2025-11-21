/**
 * Benchmark: Boundary paths with TrueForm
 *
 * Measures time to build topology structures and extract boundary paths
 * using TrueForm's make_boundary_paths.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto mesh = tf::read_stl<int>(path);
        auto polygons = mesh.polygons();

        std::size_t n_boundaries = 0;
        auto time = benchmark::min_time_of([&]() {
            // Build topology and extract boundary paths
            auto boundary_paths = tf::make_boundary_paths(polygons);
            n_boundaries = boundary_paths.size();
            benchmark::do_not_optimize(boundary_paths);
        });

        std::cout << polygons.size() << "," << time << "\n";
    }

    return 0;
}
