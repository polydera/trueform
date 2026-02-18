/**
 * Half-edge construction benchmark with CGAL - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "half_edges-cgal.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace benchmark {

int run_half_edges_cgal_benchmark(const std::vector<std::string> &mesh_paths,
                                  int n_samples, std::ostream &out) {
  using Kernel = CGAL::Simple_cartesian<float>;
  using Point_3 = Kernel::Point_3;
  using Mesh = CGAL::Surface_mesh<Point_3>;

  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);

    // Pre-convert to CGAL soup format (shared cost, not benchmarked)
    auto cgal_points = benchmark::cgal::to_cgal_points(polygons.points());
    auto cgal_faces = benchmark::cgal::to_cgal_faces(polygons.faces());

    auto time = benchmark::min_time_of(
        [&]() {
          Mesh mesh;
          PMP::polygon_soup_to_polygon_mesh(cgal_points, cgal_faces, mesh);
          benchmark::do_not_optimize(mesh);
        },
        n_samples);

    out << polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
