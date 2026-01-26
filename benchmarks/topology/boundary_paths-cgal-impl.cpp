/**
 * Boundary paths benchmark with CGAL - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boundary_paths-cgal.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

namespace benchmark {

int run_boundary_paths_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    // Convert to polygon soup format
    auto points = benchmark::cgal::to_cgal_points(r_polygons.points());
    auto faces = benchmark::cgal::to_cgal_faces(r_polygons.faces());

    std::size_t n_boundaries = 0;
    auto time = benchmark::min_time_of(
        [&]() {
          // Build Surface_mesh from polygon soup
          Surface_mesh mesh;
          PMP::polygon_soup_to_polygon_mesh(points, faces, mesh);

          // Extract boundary cycles
          std::vector<Surface_mesh::Halfedge_index> boundary_cycles;
          PMP::extract_boundary_cycles(mesh,
                                       std::back_inserter(boundary_cycles));
          n_boundaries = boundary_cycles.size();
          benchmark::do_not_optimize(boundary_cycles);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
