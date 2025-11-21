/**
 * Benchmark: Boundary paths with libigl
 *
 * Measures time to extract boundary loops for triangle meshes
 * using libigl's boundary_loop.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <igl/boundary_loop.h>

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to libigl format
        auto F = benchmark::igl::to_igl_faces(r_polygons.faces());

        auto time = benchmark::min_time_of([&]() {
            std::vector<std::vector<int>> L;
            ::igl::boundary_loop(F, L);
            benchmark::do_not_optimize(L);
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
