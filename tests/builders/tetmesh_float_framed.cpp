/**
 * @file tetmesh_float_framed.cpp
 * @brief The compiled tet mesher for int32 indices, float coordinates in a
 * tagged frame, double coordinates out
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/core/policy/frame.hpp>
#include <trueform/core/return_report.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>

namespace tf::test {

auto build_tet_mesh_framed(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                           const tf::transformation<float, 3> &transformation)
    -> double_tet_mesh_t {
  return tf::make_tet_mesh<tf::none_t, double>(
      mesh.polygons() | tf::tag(transformation), tf::return_report,
      tf::return_index_map);
}

} // namespace tf::test
