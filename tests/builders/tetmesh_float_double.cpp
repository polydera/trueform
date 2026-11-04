/**
 * @file tetmesh_float_double.cpp
 * @brief The compiled tet mesher for int32 indices, float coordinates in,
 * double coordinates out
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/tetmesh/make_tet_mesh.hpp>

namespace tf::test {

auto build_tet_mesh_double(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> double_tet_mesh_t {
  return tf::make_tet_mesh<tf::none_t, double>(mesh.polygons());
}

} // namespace tf::test
