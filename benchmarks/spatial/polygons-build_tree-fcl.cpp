/**
 * Benchmark: Polygons tree building with FCL
 *
 * Measures time to build spatial acceleration structure (BVH tree)
 * on triangle meshes of varying sizes using FCL's BVH.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

// FCL Type Definitions
using namespace benchmark::fcl;

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to FCL geometry
        auto [fcl_vertices, fcl_triangles] = benchmark::fcl::to_fcl_geometry(r_polygons);

        if (fcl_vertices.empty() || fcl_triangles.empty()) {
            continue;  // skipped due to empty or non-triangular mesh
        }

        auto time = benchmark::min_time_of([&]() {
            auto model = std::make_shared<Model>();
            model->beginModel(
                static_cast<int>(fcl_triangles.size()),
                static_cast<int>(fcl_vertices.size()));
            model->addSubModel(fcl_vertices, fcl_triangles);
            model->endModel();
            benchmark::do_not_optimize(model);
        });

        std::cout << fcl_triangles.size() << "," << time << "\n";
    }

    return 0;
}
