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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/faces.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../face_membership_like.hpp"
#include "../traversal/label_vertex_fans.hpp"
#include "../traversal/vertex_fan_is_manifold.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Where the points a vertex fan split mints begin, vertex by vertex.
///
/// Block `v` holds one fresh id per fan past the first, so a vertex that is
/// one fan — or that no fan may be separated from — holds none, and the last
/// offset is how many points the split mints. The verdict answers the whole
/// manifold mesh on its own, and only a vertex it refuses is walked again for
/// the fans themselves.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return A @ref tf::buffer of one offset per vertex plus the total.
template <typename Policy, typename Policy1>
auto make_vertex_fan_mint_offsets(const tf::faces<Policy> &faces,
                                  const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(faces[0][0])>;
  struct local_t {
    tf::buffer<tf::topology::vertex_fan_corner<Index>> corners;
  };

  auto n_points = Index(fm.size());
  tf::buffer<Index> offsets;
  offsets.allocate(std::size_t(n_points) + 1);
  offsets[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n_points),
      [&faces, &fm, &offsets](Index v, local_t &local) {
        if (tf::topology::vertex_fan_is_manifold(faces, fm, v)) {
          offsets[v + 1] = 0;
          return;
        }
        auto fans = tf::topology::label_vertex_fans(faces, fm, v, local.corners);
        offsets[v + 1] = fans - Index(fans > 0);
      },
      local_t{}, tf::checked);
  for (Index v = 0; v < n_points; ++v)
    offsets[v + 1] += offsets[v];
  return offsets;
}

} // namespace tf::topology
