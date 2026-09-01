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
#include "../../topology/cdt/constrained_delaunay_full_span_alias.hpp"
#include "./plane_member_statements.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Fold PA attribution over each exact physical constraint. Every elected CDT
/// owner then reads the same member mask, while a selective boundary override
/// changes only PA aliases and leaves generic CDT occurrence parity intact.
/// The class's verdicts are PARITIES, not presence: coincident statements
/// toggle — the mesh folding onto one carrier cancels its own wall — which is
/// the same repeated-boundary law the triangulation's own marking states.
///
/// A folded row states the block's whole member set, so its length changes:
/// the fold appends the new rows and repoints their tickets, which costs what
/// the collision touched. A member the row itself never named inherits the
/// parity alone — the SIDE is that row's own statement and no alias makes one.
/// A lone carrier names no members, so it folds no rows.
template <typename Index>
auto union_plane_constraint_aliases(
    const tf::buffer<
        tf::topology::cdt::constrained_delaunay_full_span_alias<Index>>
        &spans,
    const tf::buffer<std::array<Index, 2>> &alias_blocks,
    const tf::buffer<char> &boundaries,
    tf::buffer<std::array<Index, 2>> &cons_row,
    tf::buffer<plane_member_statement<Index>> &cons_statements,
    tf::buffer<plane_member_statement<Index>> &folded,
    tf::buffer<char> &boundary_promotions) -> void {
  boundary_promotions.clear();
  if (alias_blocks.size() == 0)
    return;
  boundary_promotions.allocate(boundaries.size());
  std::fill(boundary_promotions.begin(), boundary_promotions.end(), char(0));
  bool any_boundary = false;

  for (const auto &block : alias_blocks) {
    const auto begin = std::size_t(block[0]);
    const auto end = std::size_t(block[1]);
    bool boundary = false;
    for (auto row = begin; row < end; ++row)
      boundary = boundary !=
                 (boundaries[std::size_t(spans[row].input)] != 0);
    if (boundary) {
      any_boundary = true;
      for (auto row = begin; row < end; ++row)
        boundary_promotions[std::size_t(spans[row].input)] = char(1);
    }
    if (cons_row.size() == 0)
      continue;

    folded.clear();
    for (auto row = begin; row < end; ++row) {
      const auto stated = cons_row[std::size_t(spans[row].input)];
      for (auto at = stated[0]; at != stated[1]; ++at)
        state_plane_member_statement(
            folded, 0, Index(-1), cons_statements[std::size_t(at)].member,
            short(-1), cons_statements[std::size_t(at)].parity);
    }
    for (auto row = begin; row < end; ++row) {
      const auto input = spans[row].input;
      const auto stated = cons_row[std::size_t(input)];
      const auto rewritten = Index(cons_statements.size());
      auto at = stated[0];
      for (const auto &member : folded) {
        while (at != stated[1] &&
               cons_statements[std::size_t(at)].member < member.member)
          ++at;
        auto side = short(-1);
        if (at != stated[1] &&
            cons_statements[std::size_t(at)].member == member.member)
          side = cons_statements[std::size_t(at)].side;
        if (member.parity == char(0) && side < short(0))
          continue;
        cons_statements.push_back({member.member, input, side, member.parity});
      }
      cons_row[std::size_t(input)] = {rewritten,
                                      Index(cons_statements.size())};
    }
  }
  if (!any_boundary)
    boundary_promotions.clear();
}

} // namespace tf::arrangement
