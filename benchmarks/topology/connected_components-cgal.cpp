/**
 * Benchmark: Connected components with CGAL
 *
 * Measures time to build Surface_mesh from polygon soup and compute
 * connected component labels using CGAL's Polygon_mesh_processing.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to polygon soup format
        auto points = benchmark::cgal::to_cgal_points(r_polygons.points());
        auto faces = benchmark::cgal::to_cgal_faces(r_polygons.faces());

        std::size_t n_components = 0;
        auto time = benchmark::min_time_of([&]() {
            // Build Surface_mesh from polygon soup
            Surface_mesh mesh;
            PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);

            // Compute connected components
            std::vector<std::size_t> face_cc(num_faces(mesh));
            n_components = PMP::connected_components(mesh,
                CGAL::make_property_map(face_cc));
            benchmark::do_not_optimize(face_cc);
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
