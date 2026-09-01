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
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./gather_vertex_candidates.hpp"
#include "./place_vertex.hpp"
#include "./placement_tables.hpp"
#include "./quantized_plane.hpp"

#include <cstddef>

namespace tf::exact::door {

/// Every original vertex of every form placed on the lattice in one
/// traversal of the flat vertex space — the largest independent carrier
/// the question has, since a placement reads only the names its own
/// incident faces stated.
///
/// The product is the placed table by flat id. Nothing about a tag
/// enters here; the tables carry the offsets that made the space flat.
///
/// The cutoff is the vertex count at which the threading pays: a
/// placement searches the widest triple of its candidate list, so the
/// per-vertex cost is cubic in a valence and not a constant.
template <typename Index, typename Int, typename RealType>
auto place_vertices(const placement_tables<Index, Int, RealType> &tables,
                    Int tolerance, tf::buffer<tf::point<Int, 3>> &placed)
    -> void {
  struct local_t {
    tf::small_vector<quantized_plane<Int>, 16> candidates;
  };
  placed.allocate(tables.points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(tables.points.size())),
      [&tables, &placed, tolerance](Index flat, local_t &local) {
        const auto direction =
            gather_vertex_candidates(tables, flat, tolerance, local.candidates);
        placed[std::size_t(flat)] =
            place_vertex(tables.points[std::size_t(flat)],
                         tf::make_range(local.candidates.begin(),
                                        local.candidates.end()),
                         direction, tolerance)
                .point;
      },
      local_t{}, tf::checked(64));
}

} // namespace tf::exact::door
