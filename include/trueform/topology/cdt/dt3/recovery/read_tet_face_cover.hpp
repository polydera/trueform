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
#include "../../../../core/small_vector.hpp"
#include "../../../../exact/meta.hpp"
#include "../find_tet_edge.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_orient3d_sign.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_projected_area2.hpp"
#include "./tet_projected_point.hpp"
#include "./tet_rim_piece.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_set>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Read the facets of the complex that cover a face, named by `plane`, a
/// triangle of it, and its `rim` (@ref tet_rim_piece, sorted): from each
/// piece the facet about it on the plane on the piece's side, inside the
/// face since no facet of the complex crosses a rim piece, then across each
/// non-rim edge of a facet found the one other facet on the plane, each
/// kept once. `tickets` receives them as `4 cell + slot` from the lower of
/// a facet's two cells, `frontier` the edges — of a facet found or of the
/// rim — with no facet on the plane inside across them, slots ascending.
/// Returns twice the area covered on the projection `axes`, every site read
/// in the decision's `sites` (@ref tet_decision_sites), one depth for the
/// whole cover.
template <typename Owner, typename Sites, typename Index>
auto read_tet_face_cover(const Owner &owner, const Sites &sites,
                         const std::array<Index, 3> &plane,
                         const std::pair<int, int> &axes,
                         const tf::buffer<tet_rim_piece<Index>> &rim,
                         tf::buffer<Index> &tickets,
                         tf::buffer<std::array<Index, 2>> &frontier) ->
    typename tf::exact::meta<typename Sites::int_type>::T2 {
  using Int = typename Sites::int_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  using ref_t = tet_edge_ref<Index>;
  const auto &pa = sites[std::size_t(plane[0])].pt;
  const auto &pb = sites[std::size_t(plane[1])].pt;
  const auto &pc = sites[std::size_t(plane[2])].pt;
  const auto on_plane = [&](Index s) {
    return s != Owner::infinite &&
           tet_orient3d_sign<Int>(pa, pb, pc, sites[std::size_t(s)].pt) == 0;
  };
  const auto flat = [&sites, &axes](Index s) {
    return tet_projected_point(sites[std::size_t(s)].pt, axes);
  };
  const auto is_rim = [&rim](Index u, Index v) {
    return std::binary_search(
        rim.begin(), rim.end(),
        tet_rim_piece<Index>{std::min(u, v), std::max(u, v), 0});
  };
  const auto ticket_of = [&owner](const ref_t &handle) {
    const Index cell = handle.cell;
    const std::size_t slot = tet_edge_oppo(handle);
    const Index across = owner._neighbors[std::size_t(cell)][slot];
    if (across < cell) {
      std::size_t back = 0;
      while (owner._neighbors[std::size_t(across)][back] != cell)
        ++back;
      return Index(4 * std::size_t(across) + back);
    }
    return Index(4 * std::size_t(cell) + slot);
  };
  tickets.clear();
  frontier.clear();
  tf::small_vector<Index, 64> seen;
  std::unordered_set<Index> hashed;
  const auto admit = [&seen, &hashed](Index ticket) {
    if (seen.size() < 48) {
      if (std::find(seen.begin(), seen.end(), ticket) != seen.end())
        return false;
    } else {
      if (!hashed.size())
        hashed.insert(seen.begin(), seen.end());
      if (!hashed.insert(ticket).second)
        return false;
    }
    seen.push_back(ticket);
    return true;
  };
  T2 area(0);
  struct due_t {
    Index u, v, x;
  };
  tf::small_vector<due_t, 64> due;
  // 0: no apex about u v that `accept` takes; 1: one, its facet kept now;
  // 2: one, its facet kept before
  const auto across_plane = [&](Index u, Index v, const auto &accept,
                                Index &x) {
    ref_t handle{};
    if (!find_tet_edge(owner, u, v, handle))
      return 0;
    auto spin = handle;
    do {
      const Index apex = tet_edge_corners(owner, spin)[2];
      if (accept(apex)) {
        x = apex;
        const Index ticket = ticket_of(spin);
        if (!admit(ticket))
          return 2;
        tickets.push_back(ticket);
        return 1;
      }
      spin = tet_edge_next(owner, spin);
    } while (spin.cell != handle.cell);
    return 0;
  };
  for (const auto &piece : rim) {
    Index x = Owner::none;
    const int found = across_plane(
        piece.lo, piece.hi,
        [&](Index apex) {
          return on_plane(apex) &&
                 tet_orient2d_sign<Int>(flat(piece.lo), flat(piece.hi),
                                        flat(apex)) == piece.side;
        },
        x);
    if (found == 1)
      due.push_back({piece.lo, piece.hi, x});
    else if (found == 0)
      frontier.push_back({piece.lo, piece.hi});
  }
  for (std::size_t head = 0; head < due.size(); ++head) {
    const auto facet = due[head];
    area = area + tet_projected_area2(sites, axes, facet.u, facet.v, facet.x);
    const std::array<std::array<Index, 3>, 2> edges{
        {{facet.v, facet.x, facet.u}, {facet.x, facet.u, facet.v}}};
    for (const auto &edge : edges) {
      if (is_rim(edge[0], edge[1]))
        continue;
      Index y = Owner::none;
      const int found = across_plane(
          edge[0], edge[1],
          [&](Index apex) { return apex != edge[2] && on_plane(apex); }, y);
      if (found == 1)
        due.push_back({edge[0], edge[1], y});
      else if (found == 0)
        frontier.push_back({std::min(edge[0], edge[1]),
                            std::max(edge[0], edge[1])});
    }
  }
  std::sort(frontier.begin(), frontier.end());
  frontier.erase_till_end(std::unique(frontier.begin(), frontier.end()));
  return area;
}

} // namespace tf::topology::cdt::dt3
