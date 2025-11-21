/**
 * Benchmark: Connected components with TrueForm
 *
 * Measures time to build topology structures and compute connected
 * component labels using TrueForm's label_connected_components.
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

        short n_components = 0;
        auto time = benchmark::min_time_of([&]() {
            // no precompute of topology
            auto [labels, n_components] = tf::make_manifold_edge_connected_component_labels<int>(polygons);
            benchmark::do_not_optimize(labels);
        });

        std::cout << polygons.size() << "," << time << "\n";
    }

    return 0;
}
