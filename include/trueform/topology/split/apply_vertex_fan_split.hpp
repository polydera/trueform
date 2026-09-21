/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/faces.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/take.hpp"
#include "../face_membership_like.hpp"
#include "../traversal/label_vertex_fans.hpp"
#include <cstddef>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Move every fan past the first onto a minted vertex of its own.
///
/// The fan keeping the vertex writes nothing, so `out_faces` enters as the
/// source winding and leaves with only the moved corners rewritten.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @tparam Index The integer type for indices.
/// @tparam Policy2 The output faces policy type.
/// @param faces The source faces range.
/// @param fm The face membership structure.
/// @param mint_offsets Where each vertex's minted ids begin.
/// @param out_faces A copy of the source faces, rewired in place.
/// @return The point map: for each output point the source point it copies.
template <typename Policy, typename Policy1, typename Index, typename Policy2>
auto apply_vertex_fan_split(const tf::faces<Policy> &faces,
                            const tf::face_membership_like<Policy1> &fm,
                            const tf::buffer<Index> &mint_offsets,
                            tf::faces<Policy2> out_faces) -> tf::buffer<Index> {
  auto n_points = Index(fm.size());
  auto n_mints = mint_offsets[n_points];
  tf::buffer<Index> point_map;
  point_map.allocate(std::size_t(n_points) + std::size_t(n_mints));
  tf::parallel_iota(tf::take(point_map, std::size_t(n_points)), Index(0));
  if (!n_mints)
    return point_map;

  struct local_t {
    tf::buffer<tf::topology::vertex_fan_corner<Index>> corners;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(n_points),
      [&faces, &fm, &mint_offsets, &point_map, &out_faces, n_points](
          Index v, local_t &local) {
        auto base = mint_offsets[v];
        auto end = mint_offsets[v + 1];
        if (base == end)
          return;
        tf::topology::label_vertex_fans(faces, fm, v, local.corners);
        for (const auto &corner : local.corners)
          if (corner.fan > 0)
            out_faces[corner.face][corner.corner] =
                n_points + base + corner.fan - 1;
        for (Index m = base; m < end; ++m)
          point_map[std::size_t(n_points) + std::size_t(m)] = v;
      },
      local_t{}, tf::checked);
  return point_map;
}

} // namespace tf::topology
