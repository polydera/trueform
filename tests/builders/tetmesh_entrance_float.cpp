/**
 * @file tetmesh_entrance_float.cpp
 * @brief The compiled tet entrance for int32 indices, float coordinates,
 * quantized onto the int32 lattice at the input's precision
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tetmesh_builders.hpp"

#include <trueform/core/polygons.hpp>
#include <trueform/tetmesh/entrance/make_tet_entrance.hpp>

#include <cstdint>

namespace tf::test {

auto build_tet_entrance(const tf::points_buffer<float, 3> &points,
                        const tf::blocked_buffer<int, 3> &faces)
    -> float_tet_entrance_t {
  const auto polygons = tf::make_polygons(faces, points.points());
  return tf::tetmesh::make_tet_entrance<int, tf::exact::int32, std::int64_t>(
      polygons);
}

} // namespace tf::test
