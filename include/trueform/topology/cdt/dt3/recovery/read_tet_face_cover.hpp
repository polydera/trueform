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
#include "../../../../core/buffer.hpp"
#include "./spread_tet_face_cover.hpp"
#include "./state_tet_cover_layers.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_rim_piece.hpp"
#include <algorithm>
#include <array>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Read the facets of the complex that cover a face, named by `plane`, a
/// triangle of it, and its `rim` (@ref tet_rim_piece, sorted): the cover
/// spread from every piece of the rim (@ref spread_tet_face_cover), inside
/// the face since no facet of the complex crosses a rim piece. `tickets`
/// receives its facets, `frontier` the rim of what it leaves, sorted and
/// each piece once — a rim naming one piece twice, as two triangles of one
/// face running one edge alike do, leaves it once: the
/// cover tiles the face exactly when the frontier is empty, and then
/// `layers` receives, aligned with `tickets`, how many of the face's own
/// triangles lie over each facet (@ref state_tet_cover_layers), empty
/// otherwise. Every site is
/// read in the decision's `sites` (@ref tet_decision_sites) on the
/// projection `axes`, one depth for the whole cover; the spread's own
/// lists are the caller's (@ref tet_face_cover_scratch).
template <typename Owner, typename Sites, typename Index>
auto read_tet_face_cover(const Owner &owner, const Sites &sites,
                         const std::array<Index, 3> &plane,
                         const std::pair<int, int> &axes,
                         const tf::buffer<tet_rim_piece<Index>> &rim,
                         tf::buffer<Index> &tickets,
                         tf::buffer<tet_rim_piece<Index>> &frontier,
                         tf::buffer<Index> &layers,
                         tet_face_cover_scratch<Index> &scratch) -> void {
  tickets.clear();
  frontier.clear();
  spread_tet_face_cover(owner, sites, plane, axes, rim, tickets, frontier,
                        scratch);
  std::sort(frontier.begin(), frontier.end(), tet_rim_piece_precedes<Index>);
  frontier.erase_till_end(std::unique(
      frontier.begin(), frontier.end(),
      [](const tet_rim_piece<Index> &a, const tet_rim_piece<Index> &b) {
        return a.lo == b.lo && a.hi == b.hi && a.side == b.side;
      }));
  layers.clear();
  if (!frontier.size())
    state_tet_cover_layers(sites, axes, rim, layers, scratch);
}

} // namespace tf::topology::cdt::dt3
