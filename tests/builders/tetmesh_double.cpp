/**
 * @file tetmesh_double.cpp
 * @brief The compiled tet mesher for int32 indices, double coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/core/return_report.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>

namespace tf::test {

auto build_tet_mesh(const tf::polygons_buffer<int, double, 3, 3> &mesh)
    -> double_tet_mesh_t {
  return tf::make_tet_mesh(mesh.polygons(), tf::return_report,
                           tf::return_index_map);
}

} // namespace tf::test
