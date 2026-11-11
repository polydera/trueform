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
#include "../find_tet_edge.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_facet_ticket.hpp"
#include "../tet_orient3d_sign.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_projected_point.hpp"
#include "./tet_rim_piece.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Spread a face's cover, named by `plane`, a triangle of it, from the
/// pieces `entries` (sorted): across each the facet of the complex on the
/// plane on the piece's side, then across each edge of a facet found that
/// `entries` does not name the one other facet on the plane, each kept
/// once. A facet found is appended to `tickets` as `4 cell + slot` from the
/// lower of its two cells, an edge with no facet on the plane across it to
/// `frontier` with the side the uncovered face lies on. The tickets it has
/// taken, the facets it owes their edges and the stars it finds them
/// through are the caller's own lists (@ref tet_face_cover_scratch). The
/// facet at position `p` of `tickets` is `scratch.due[p]`, read as its
/// sides `u v`, `v x`, `x u`, and `scratch.adjacent[p]` names the facet of
/// the cover across each, `-1` for none and `-2` for an entry, across which
/// the spread does not walk. Every site is read in the decision's `sites` (@ref
/// tet_decision_sites) on the projection `axes`.
template <typename Owner, typename Sites, typename Index>
auto spread_tet_face_cover(const Owner &owner, const Sites &sites,
                           const std::array<Index, 3> &plane,
                           const std::pair<int, int> &axes,
                           const tf::buffer<tet_rim_piece<Index>> &entries,
                           tf::buffer<Index> &tickets,
                           tf::buffer<tet_rim_piece<Index>> &frontier,
                           tet_face_cover_scratch<Index> &scratch) -> void {
  using Int = typename Sites::int_type;
  using ref_t = tet_edge_ref<Index>;
  const auto pa = sites[std::size_t(plane[0])].pt;
  const auto pb = sites[std::size_t(plane[1])].pt;
  const auto pc = sites[std::size_t(plane[2])].pt;
  const auto on_plane = [&](Index s) {
    return s != Owner::infinite &&
           tet_orient3d_sign<Int>(pa, pb, pc, sites[std::size_t(s)].pt) == 0;
  };
  const auto flat = [&sites, &axes](Index s) {
    return tet_projected_point(sites[std::size_t(s)].pt, axes);
  };
  const auto is_entry = [&entries](Index u, Index v) {
    return std::binary_search(
        entries.begin(), entries.end(),
        tet_rim_piece<Index>{std::min(u, v), std::max(u, v), 0});
  };
  auto &hashed = scratch.hashed;
  auto &seen = scratch.seen;
  auto &star = scratch.star;
  auto &due = scratch.due;
  auto &adjacent = scratch.adjacent;
  seen.clear();
  due.clear();
  adjacent.clear();
  // the position of `ticket` among the ones taken, `-1` for a new one, which
  // is taken
  const auto admit = [&seen, &hashed](Index ticket) {
    const Index next = Index(seen.size());
    if (seen.size() < 48) {
      const auto at = std::find(seen.begin(), seen.end(), ticket);
      if (at != seen.end())
        return Index(at - seen.begin());
    } else {
      if (seen.size() == 48) {
        hashed.clear();
        for (std::size_t i = 0; i < seen.size(); ++i)
          hashed.insert({seen[i], Index(i)});
      }
      const auto placed = hashed.insert({ticket, next});
      if (!placed.second)
        return placed.first->second;
    }
    seen.push_back(ticket);
    return Index(-1);
  };
  // 0: no apex about u v that `accept` takes; 1: one, its facet appended;
  // 2: one, its facet already admitted, at `at`
  const auto across_plane = [&](Index u, Index v, const auto &accept, Index &x,
                                Index &at) {
    ref_t handle{};
    if (!find_tet_edge(owner, u, v, handle, star))
      return 0;
    auto spin = handle;
    do {
      const Index apex = tet_edge_corners(owner, spin)[2];
      if (accept(apex)) {
        x = apex;
        const Index ticket = tet_facet_ticket(owner, spin);
        at = admit(ticket);
        if (at != Index(-1))
          return 2;
        tickets.push_back(ticket);
        return 1;
      }
      spin = tet_edge_next(owner, spin);
    } while (spin.cell != handle.cell);
    return 0;
  };
  // a facet's sides as its record names them, `u v`, `v x`, `x u`: the facet
  // of the cover across each, `-1` for none, `-2` for an entry it was not
  // crossed through
  for (const auto &piece : entries) {
    Index x = Owner::none, at = Owner::none;
    const int found = across_plane(
        piece.lo, piece.hi,
        [&](Index apex) {
          return on_plane(apex) &&
                 tet_orient2d_sign<Int>(flat(piece.lo), flat(piece.hi),
                                        flat(apex)) == piece.side;
        },
        x, at);
    if (found == 1) {
      due.push_back({piece.lo, piece.hi, x});
      adjacent.push_back({Index(-2), Index(-1), Index(-1)});
    } else if (found == 0)
      frontier.push_back(piece);
  }
  for (std::size_t head = 0; head < due.size(); ++head) {
    const auto facet = due[head];
    const std::array<std::array<Index, 3>, 2> edges{
        {{facet.v, facet.x, facet.u}, {facet.x, facet.u, facet.v}}};
    for (std::size_t side = 1; side < 3; ++side) {
      const auto &edge = edges[side - 1];
      if (is_entry(edge[0], edge[1])) {
        adjacent[head][side] = Index(-2);
        continue;
      }
      Index y = Owner::none, at = Owner::none;
      const int found = across_plane(
          edge[0], edge[1],
          [&](Index apex) { return apex != edge[2] && on_plane(apex); }, y, at);
      if (found == 2)
        adjacent[head][side] = at;
      if (found == 1) {
        adjacent[head][side] = Index(due.size());
        due.push_back({edge[0], edge[1], y});
        adjacent.push_back({Index(head), Index(-1), Index(-1)});
      } else if (found == 0) {
        const Index lo = std::min(edge[0], edge[1]);
        const Index hi = std::max(edge[0], edge[1]);
        frontier.push_back(
            {lo, hi,
             -tet_orient2d_sign<Int>(flat(lo), flat(hi), flat(edge[2]))});
      }
    }
  }
}

} // namespace tf::topology::cdt::dt3
