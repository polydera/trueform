/**
 * Polygons tree building CGAL - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons-build_tree-cgal.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>

using namespace benchmark::cgal;

// CGAL Type Definitions
using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
using Traits = CGAL::AABB_traits_3<Kernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

namespace benchmark {

int run_polygons_build_tree_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "bv,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    auto mesh = benchmark::cgal::to_cgal_mesh(r_polygons);

    auto time = benchmark::min_time_of(
        [&]() {
          Tree tree(faces(mesh).begin(), faces(mesh).end(), mesh);
          tree.build();
          benchmark::do_not_optimize(tree);
        },
        n_samples);

    out << "AABB," << r_polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
