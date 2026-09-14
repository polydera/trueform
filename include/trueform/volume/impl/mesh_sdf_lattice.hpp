/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include "../../core/aabb.hpp"
#include "../../core/aabb_from.hpp"
#include "../../core/aabb_union.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/transformed.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../spatial/aabb_from.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief The mesh and the grid on one lattice: the vertices converted
/// through a converter spanning both, and the samples' per-axis lattice
/// coordinates in ascending order — `flips[d]` says the grid's axis `d`
/// runs opposite to it.
template <typename Int> struct mesh_sdf_lattice {
  tf::buffer<tf::point<Int, 3>> points;
  std::array<tf::buffer<Int>, 3> axes;
  std::array<bool, 3> flips;
};

template <typename Int, typename InReal, typename Polygons, typename P0,
          typename P1>
auto make_mesh_sdf_lattice(const Polygons &polygons,
                           const std::array<int, 3> &dims,
                           const tf::point_like<3, P0> &spacing,
                           const tf::point_like<3, P1> &origin)
    -> mesh_sdf_lattice<Int> {
  mesh_sdf_lattice<Int> lattice;
  const auto pose = tf::frame_of(polygons);
  tf::point<InReal, 3> g0, g1;
  for (std::size_t d = 0; d < 3; ++d) {
    const auto a = static_cast<InReal>(origin[d]);
    const auto b = a + static_cast<InReal>(std::max(dims[d] - 1, 0)) *
                           static_cast<InReal>(spacing[d]);
    g0[d] = std::min(a, b);
    g1[d] = std::max(a, b);
  }
  const auto box = tf::aabb_union(
      tf::make_aabb(g0, g1),
      tf::transformed(tf::aabb_from(polygons.tree()), pose));
  const auto conv = tf::exact::make_pt_converter<Int, InReal>(box);

  lattice.points.allocate(polygons.points().size());
  tf::parallel_transform(polygons.points(), lattice.points,
                         [&](const auto &p) {
                           return conv(tf::transformed(p, pose));
                         });

  for (std::size_t d = 0; d < 3; ++d) {
    auto &axis = lattice.axes[d];
    axis.allocate(std::size_t(dims[d]));
    for (int i = 0; i < dims[d]; ++i) {
      tf::point<InReal, 3> p{static_cast<InReal>(origin[0]),
                             static_cast<InReal>(origin[1]),
                             static_cast<InReal>(origin[2])};
      p[d] += static_cast<InReal>(i) * static_cast<InReal>(spacing[d]);
      axis[std::size_t(i)] = conv(p)[d];
    }
    lattice.flips[d] = dims[d] > 1 && axis[0] > axis[std::size_t(dims[d]) - 1];
    if (lattice.flips[d])
      std::reverse(axis.begin(), axis.end());
  }
  return lattice;
}

} // namespace volume_detail
} // namespace tf
