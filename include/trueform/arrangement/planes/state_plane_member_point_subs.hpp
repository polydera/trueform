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
#include "../../topology/topo_id.hpp"
#include "../../topology/topo_type.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// CORE. Where ONE point sits on ONE member's polygon, given a side that
/// member states through it.
///
/// The member's own sides are the whole authority: a point an own-side
/// constraint ends lies on that side, and a point two DIFFERENT sides end is
/// the corner where they meet — the later one in loop order, which the
/// wrapping pair `(0, n - 1)` states as corner `0`. Everything else is
/// interior, and so is every point of a plane the member states no side on.
///
/// `touched` collects each point the first time it leaves that interior
/// default, so a table shared across a stack's members is reset by walking
/// what was stated.
template <typename Index>
auto elect_plane_point_sub(Index point, short side,
                           tf::buffer<tf::topo_id<short>> &subs,
                           tf::buffer<Index> &touched) -> void {
  auto &sub = subs[std::size_t(point)];
  if (sub.label == tf::topo_type::vertex)
    return;
  if (sub.label != tf::topo_type::edge) {
    touched.push_back(point);
    sub = {side, tf::topo_type::edge};
    return;
  }
  if (sub.id == side)
    return;
  const auto lo = std::min(sub.id, side);
  const auto hi = std::max(sub.id, side);
  sub = {short(hi == lo + short(1) ? hi : lo), tf::topo_type::vertex};
}

/// CORE. A lone carrier's whole point table: its one member states a side per
/// constraint, so the constraints ARE the walk.
template <typename Index>
auto state_plane_point_subs(const tf::buffer<Index> &cons,
                            const tf::buffer<short> &cons_side,
                            std::size_t n_points,
                            tf::buffer<tf::topo_id<short>> &subs,
                            tf::buffer<Index> &touched) -> void {
  subs.allocate(n_points);
  std::fill(subs.begin(), subs.end(),
            tf::topo_id<short>{short(0), tf::topo_type::face});
  touched.clear();
  const auto n_cons = cons.size() / 2;
  for (std::size_t constraint = 0; constraint < n_cons; ++constraint) {
    const auto side = cons_side[constraint];
    if (side < short(0))
      continue;
    elect_plane_point_sub(cons[constraint * 2], side, subs, touched);
    elect_plane_point_sub(cons[constraint * 2 + 1], side, subs, touched);
  }
}

/// CORE. One member of a stack, on the table the previous member left: the
/// statements are the member's OWN rows, ascending, so the walk costs its own
/// boundary and the reset costs what that boundary stated.
template <typename Index, typename Statements>
auto state_plane_member_point_subs(const tf::buffer<Index> &cons,
                                   const Statements &statements,
                                   tf::buffer<tf::topo_id<short>> &subs,
                                   tf::buffer<Index> &touched) -> void {
  for (const auto point : touched)
    subs[std::size_t(point)] = {short(0), tf::topo_type::face};
  touched.clear();
  for (const auto &stated : statements) {
    if (stated.side < short(0))
      continue;
    const auto constraint = std::size_t(stated.row);
    elect_plane_point_sub(cons[constraint * 2], stated.side, subs, touched);
    elect_plane_point_sub(cons[constraint * 2 + 1], stated.side, subs, touched);
  }
}

} // namespace tf::arrangement
