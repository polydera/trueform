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

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// ONE MEMBER'S STATEMENT about one constraint row of a pooled carrier: the
/// side of that member's own polygon the row lies on (`-1` for none) and
/// whether the member's statements leave a wall standing there.
///
/// A row names only the members that speak for it, so the table is the facts
/// and nothing else — a member absent from a row is silent about it on both
/// counts, which is what a dense (row x member) matrix spends its whole size
/// saying. The rows themselves are reached through a ticket, so the same
/// records serve row-major, where the coverage flood reads a wall's toggles,
/// and member-major, where a member reads its own boundary.
template <typename Index> struct plane_member_statement {
  Index member;
  Index row;
  short side;
  char parity;
};

/// CORE. Add one statement to the row being built at `begin`. Rows stay
/// ascending in member, so the flood's walk across a wall is a merge; a member
/// RESTATING the row toggles its wall — the mesh folding onto one carrier
/// cancels — and the first side it names is the one it keeps.
template <typename Index>
auto state_plane_member_statement(
    tf::buffer<plane_member_statement<Index>> &statements, std::size_t begin,
    Index row, Index member, short side, char parity) -> void {
  auto at = statements.size();
  while (at > begin && member < statements[at - 1].member)
    --at;
  if (at > begin && statements[at - 1].member == member) {
    auto &stated = statements[at - 1];
    stated.parity = char(stated.parity ^ parity);
    if (stated.side < short(0))
      stated.side = side;
    return;
  }
  statements.push_back({});
  for (auto k = statements.size() - 1; k > at; --k)
    statements[k] = statements[k - 1];
  statements[at] = {member, row, side, parity};
}

/// CORE. The same statements by MEMBER. Counts plus one prefix put each
/// member's own boundary in a disjoint run, ascending in row, which is the
/// order the point subs are elected in.
template <typename Index>
auto order_plane_member_statements(
    const tf::buffer<plane_member_statement<Index>> &statements,
    const tf::buffer<std::array<Index, 2>> &cons_row, std::size_t n_rows,
    std::size_t n_members, tf::buffer<Index> &offsets,
    tf::buffer<Index> &cursor,
    tf::buffer<plane_member_statement<Index>> &ordered) -> void {
  offsets.allocate(n_members + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (std::size_t r = 0; r < n_rows; ++r) {
    const auto row = cons_row[r];
    for (auto at = row[0]; at != row[1]; ++at)
      ++offsets[std::size_t(statements[std::size_t(at)].member) + 1];
  }
  for (std::size_t m = 1; m <= n_members; ++m)
    offsets[m] += offsets[m - 1];
  cursor.allocate(n_members);
  std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
  ordered.allocate(std::size_t(offsets[n_members]));
  for (std::size_t r = 0; r < n_rows; ++r) {
    const auto row = cons_row[r];
    for (auto at = row[0]; at != row[1]; ++at) {
      const auto &stated = statements[std::size_t(at)];
      ordered[std::size_t(cursor[std::size_t(stated.member)]++)] = stated;
    }
  }
}

} // namespace tf::arrangement
