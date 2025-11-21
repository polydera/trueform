/**
 * Benchmark: Polygons tree building with CGAL
 *
 * Measures time to build spatial acceleration structure (AABB tree)
 * on triangle meshes of varying sizes using CGAL's AABB tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

// CGAL Type Definitions
using namespace benchmark::cgal;
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits = CGAL::AABB_traits<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to CGAL mesh
        auto mesh = benchmark::cgal::to_cgal_mesh(r_polygons);

        auto time = benchmark::min_time_of([&]() {
            Tree tree(faces(mesh).begin(), faces(mesh).end(), mesh);
            tree.build();
            benchmark::do_not_optimize(tree);
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
