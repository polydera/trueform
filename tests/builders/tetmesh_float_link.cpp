/**
 * @file tetmesh_float_link.cpp
 * @brief The compiled tet mesher for int32 indices, float coordinates in a
 * form tagged with its manifold edge link alone, double out
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/core/return_report.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::test {

auto build_tet_mesh_with_link(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                              const tf::manifold_edge_link<int, 3> &link)
    -> double_tet_mesh_t {
  return tf::make_tet_mesh<tf::none_t, double>(
      mesh.polygons() | tf::tag(link), tf::return_report, tf::return_index_map);
}

} // namespace tf::test
