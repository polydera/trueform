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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/door/place_door_vertices.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/cdt/dt3/tet_position.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// A landing proposed: the member side it stands on in slots and the point
/// that stands there.
template <typename Index> struct tet_reach_proposal {
  std::array<Index, 2> side;
  Index from;
};

/// The scratch @ref land_tet_reach_points walks on.
template <typename Index> struct tet_reach_landing_workspace {
  tf::buffer<tet_reach_proposal<Index>> proposals;
};

/// THE LANDING: a point the drain states on a member's side whose nearest
/// lattice point on another side of that member, inside that side's span
/// (@ref tf::exact::door::project_onto_edge), shares its cell of the grid
/// of side `reach` (@ref tf::topology::cdt::dt3::tet_shares_reach_cell)
/// stands on that side too: `{a, b, point}` joins `edge_points` (sorted on
/// return), the point's one identity stated on a second side at that side's
/// own parameter (@ref state_tet_side_parameters). Nothing is minted or
/// moved, and nothing refuses at a landing. `edge_key(u, k)` and
/// `corners_of(u)` read a member's sides in slots and its corners as
/// identities. O(the edge points × their holders).
template <typename Int, typename Index, typename Live, typename Holds,
          typename CornersOf, typename EdgeKey>
auto land_tet_reach_points(const tf::points_buffer<Int, 3> &points,
                           tf::buffer<std::array<Index, 3>> &edge_points,
                           const tet_slot_holders<Index> &holders,
                           const Live &live, const Holds &holds,
                           const CornersOf &corners_of, const EdgeKey &edge_key,
                           Int reach, tet_reach_landing_workspace<Index> &work)
    -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using pt_t = tf::exact::pt3<Int>;
  using proposal_t = tet_reach_proposal<Index>;
  const auto placed = [&points](Index p) {
    return pt_t(points[std::size_t(p)]);
  };
  struct local_t {
    tf::buffer<proposal_t> proposals;
  };
  work.proposals.clear();
  // the proposals aggregate in input-block order, so the naming mints in
  // one order at any worker count
  // a proposal of `from` onto side `k` of member `u`, where its nearest
  // lattice point on the side, inside the span, shares its cell
  const auto propose = [&](Index u, std::size_t k, Index from,
                           const pt_t &origin, local_t &local) {
    const auto corners = corners_of(u);
    const Index a = corners[k], b = corners[(k + 1) % 3];
    if (from == a || from == b || from == corners[(k + 2) % 3])
      return;
    const auto pa = placed(a), pb = placed(b);
    T2 length2(0), along(0);
    for (std::size_t d = 0; d < 3; ++d) {
      const T1 side = T1(pb[d]) - T1(pa[d]);
      const T1 to = T1(origin[d]) - T1(pa[d]);
      length2 = length2 + T2(side) * T2(side);
      along = along + T2(side) * T2(to);
    }
    if (!(T2(0) < along) || !(along < length2))
      return;
    const auto on = tf::exact::door::project_onto_edge<Int>(origin, pa, pb);
    const pt_t landing{Int(on[0]), Int(on[1]), Int(on[2])};
    if (!tf::topology::cdt::dt3::tet_shares_reach_cell(
            tf::topology::cdt::dt3::tet_position<Int>{origin},
            tf::topology::cdt::dt3::tet_position<Int>{landing}, reach))
      return;
    local.proposals.push_back({edge_key(u, k), from});
  };
  const auto gather = [](const local_t &local,
                         tf::buffer<proposal_t> &proposals) {
    for (const auto &proposal : local.proposals)
      proposals.push_back(proposal);
  };
  // the proposals aggregate in input-block order, so the naming mints in
  // one order at any worker count
  tf::blocked_reduce_sequenced_aggregate(
      edge_points, work.proposals, local_t{},
      [&](const auto &block, local_t &local) {
        local.proposals.clear();
        for (const auto &entry : block) {
          const Index from = entry[2];
          const pt_t origin = placed(from);
          for_each_tet_slot_holder(holders, entry[0], [&](Index u) {
            if (!live(std::size_t(u)) || !holds(u, entry[1]))
              return;
            for (std::size_t k = 0; k < 3; ++k) {
              const auto key = edge_key(u, k);
              if (key[0] != entry[0] || key[1] != entry[1])
                propose(u, k, from, origin, local);
            }
          });
        }
      },
      gather, tf::checked);
  for (const auto &proposal : work.proposals)
    edge_points.push_back({proposal.side[0], proposal.side[1], proposal.from});
  std::sort(edge_points.begin(), edge_points.end());
  edge_points.erase_till_end(
      std::unique(edge_points.begin(), edge_points.end()));
}

} // namespace tf::tetmesh
