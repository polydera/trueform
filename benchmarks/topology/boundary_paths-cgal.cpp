/**
 * Benchmark: Boundary paths with CGAL
 *
 * Measures time to build Surface_mesh from polygon soup and extract
 * boundary cycles using CGAL's Polygon_mesh_processing.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to polygon soup format
        auto points = benchmark::cgal::to_cgal_points(r_polygons.points());
        auto faces = benchmark::cgal::to_cgal_faces(r_polygons.faces());

        std::size_t n_boundaries = 0;
        auto time = benchmark::min_time_of([&]() {
            // Build Surface_mesh from polygon soup
            Surface_mesh mesh;
            PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);

            // Extract boundary cycles
            std::vector<Surface_mesh::Halfedge_index> boundary_cycles;
            PMP::extract_boundary_cycles(mesh, std::back_inserter(boundary_cycles));
            n_boundaries = boundary_cycles.size();
            benchmark::do_not_optimize(boundary_cycles);
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
