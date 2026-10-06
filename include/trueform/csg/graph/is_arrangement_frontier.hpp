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

#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Whether side `s` of exposed triangle `e` lies on a piece no other
///        live triangle names — where the arrangement states no neighbour.
///
/// A filler diagonal lies on no piece and is never a frontier.
template <typename Arrangement, typename Index>
auto is_arrangement_frontier(const Arrangement &arrangement, Index e, int s)
    -> bool {
  const auto piece = arrangement.global()
                         .exposed_parent_of()[std::size_t(e) * 3 +
                                              std::size_t(s)];
  if (piece == Index(-1))
    return false;
  auto exposed_of_row = arrangement.exposed_of_row();
  auto dead = arrangement.dead();
  for (const auto row :
       arrangement.piece_incidence().rows_of_piece[std::size_t(piece)]) {
    const auto peer = exposed_of_row[std::size_t(row / Index(3))];
    if (peer == Index(-1) || peer == e || dead[peer])
      continue;
    return false;
  }
  return true;
}

} // namespace tf::csg::graph
