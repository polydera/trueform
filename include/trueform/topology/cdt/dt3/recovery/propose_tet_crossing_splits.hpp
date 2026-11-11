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
#include "../../../../exact/meta.hpp"
#include "../tet_deep_lattice.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_position.hpp"
#include "../tet_with_sites.hpp"
#include "../tetrahedralization_owner.hpp"
#include "./tet_piece_row.hpp"
#include "./tet_segment_crossing_numerator.hpp"
#include "./tet_segment_point.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_subsegment_split.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// State the two splits that resolve row `k` crossing the piece from
/// `crossed[0]` to `crossed[1]` of another segment, once in `round`, into
/// `splits`: the crossing is one position both segments' frames place
/// exactly, strictly inside each row, and each row splits there on its own
/// frame: the crossed piece first, an edge of the complex the split divides,
/// then `k`'s row, landing on the site the first minted, which the settle
/// shares (@ref settle_tet_split_sites); a position a site already holds is
/// that site on both rows. A crossing either frame cannot place
/// exactly bends a chain off its face's plane and is no split: returns
/// false there, and where the piece is no row, the rows are one segment's,
/// a row is off its frame, the lines are parallel, or the position stands
/// deeper than the lattice admits.
template <typename Int, typename Index, typename Coord>
auto propose_tet_crossing_splits(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    tet_segment_recovery<Index, Coord> &recovery, Index k,
    const std::array<Index, 2> &crossed, Index round,
    tf::buffer<tet_subsegment_split<Index, Coord>> &splits) -> bool {
  const Index other = tet_piece_row(recovery, crossed[0], crossed[1]);
  if (other == Index(-1))
    return false;
  const auto &rows = recovery.subsegments;
  const auto &row = rows[std::size_t(k)];
  if (rows[std::size_t(other)].segment == row.segment)
    return false;
  if (row.n0 < 0 || row.n1 < 0 || row.n1 - row.n0 < 2)
    return false;
  const auto &piece = rows[std::size_t(other)];
  if (piece.n0 < 0 || piece.n1 < 0 || piece.n1 - piece.n0 < 2)
    return false;
  const auto placed = [&owner, &recovery, &rows](const auto &on, Index c,
                                                  Index d, std::int64_t &numerator,
                                                  tet_position<Coord> &position) {
    const std::size_t s = std::size_t(on.segment);
    const Index head = rows[std::size_t(recovery.first[s])].a;
    const Index tail = rows[std::size_t(recovery.last[s])].b;
    const int bits = recovery.bits[s];
    numerator = tet_segment_crossing_numerator(owner, head, tail, c, d, bits);
    if (numerator <= on.n0 || numerator >= on.n1)
      return false;
    position = tet_with_sites(
        owner, tet_deepest_of(owner, head, tail), [&](const auto &sites) {
          using lattice = typename std::decay_t<decltype(sites)>::int_type;
          return tet_minted_position<Coord>(
              tet_segment_point<typename tf::exact::meta<lattice>::T2>(
                  sites[std::size_t(head)].pt, sites[std::size_t(tail)].pt,
                  numerator, bits),
              0, sites.depth + bits);
        });
    return !position.depth ||
           (recovery.deepens &&
            position.depth <= tet_deep_lattice<Int, Coord>::cap);
  };
  std::int64_t numerator = -1, numerator_piece = -1;
  tet_position<Coord> position{}, position_piece{};
  const bool a_ok = placed(row, crossed[0], crossed[1], numerator, position);
  const bool b_ok = placed(piece, row.a, row.b, numerator_piece, position_piece);
  if (!a_ok || !b_ok)
    return false;
  if (position.depth != position_piece.depth ||
      position.floor != position_piece.floor ||
      position.residual != position_piece.residual)
    return false;
  recovery.proposed[std::size_t(k)] = round;
  recovery.proposed[std::size_t(other)] = round;
  splits.push_back({other, numerator_piece, position, Index(-1)});
  splits.push_back({k, numerator, position, Index(-1)});
  return true;
}

} // namespace tf::topology::cdt::dt3
