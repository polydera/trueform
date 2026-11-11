/**
 * @file tetmesh_float_tree.cpp
 * @brief The compiled tet mesher for int32 indices, float coordinates in a
 * form tagged with `tree`, double out
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/core/return_report.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>

namespace tf::test {

auto build_tet_mesh_with_tree(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                              const tf::aabb_tree<int, float, 3> &tree)
    -> double_tet_mesh_t {
  return tf::make_tet_mesh<tf::none_t, double>(
      mesh.polygons() | tf::tag(tree), tf::return_report, tf::return_index_map);
}

} // namespace tf::test
