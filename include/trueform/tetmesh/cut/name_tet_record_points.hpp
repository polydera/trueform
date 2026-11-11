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
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../exact/vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Name every point of the pair records `records` in `point_of`: a vertex
/// anchor is its surface point `global` names, and a created point the
/// identity the drain's `authority` names at its lattice position
/// (`positions`) for its pair (`pairs` by created id), every identity it
/// mints appended to `points` at its position, in mint order — so a point
/// of `points` is the identity of its index. `ids` is scratch.
/// O(record points log record points).
template <typename Index, typename Int, typename Records, typename Authority>
auto name_tet_record_points(tf::points_buffer<Int, 3> &points,
                            const Records &records,
                            const tf::buffer<Index> &global,
                            const tf::buffer<tf::exact::pt3<Int>> &positions,
                            const tf::buffer<std::array<Index, 2>> &pairs,
                            Authority &authority, tf::buffer<Index> &point_of,
                            tf::buffer<Index> &ids) -> void {
  const auto n_vertex = std::size_t(records.n_vertex_points());
  const auto n_ids = std::size_t(records.n_points());
  point_of.allocate(n_ids);
  for (std::size_t id = 0; id < n_vertex; ++id)
    point_of[id] = global[std::size_t(records.vertex_anchor(Index(id)).vid)];
  const Index first = authority.name_positions(
      tf::make_range(positions.begin() + std::ptrdiff_t(n_vertex),
                     positions.end()),
      pairs, ids);
  for (std::size_t id = n_vertex; id < n_ids; ++id)
    point_of[id] = ids[id - n_vertex];
  for (Index id = first; id < authority.n_identities(); ++id)
    points.push_back(authority.position_of(id));
}

} // namespace tf::tetmesh
