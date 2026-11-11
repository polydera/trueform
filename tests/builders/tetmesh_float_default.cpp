/**
 * @file tetmesh_float_default.cpp
 * @brief The compiled tet mesher's default call on int32 indices and float
 * coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/tetmesh/make_tet_mesh.hpp>

namespace tf::test {

auto build_tet_mesh_default(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> default_tet_mesh_t {
  return tf::make_tet_mesh(mesh.polygons());
}

} // namespace tf::test
