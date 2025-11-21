/**
 * Benchmark: Connected components with libigl
 *
 * Measures time to compute connected component labels for triangle meshes
 * using libigl's facet_components.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <igl/facet_components.h>

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to libigl format
        auto F = benchmark::igl::to_igl_faces(r_polygons.faces());

        auto time = benchmark::min_time_of([&]() {
            Eigen::VectorXi C;
            ::igl::facet_components(F, C);
            benchmark::do_not_optimize(C);
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
