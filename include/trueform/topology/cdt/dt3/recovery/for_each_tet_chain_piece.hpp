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
#include "../../../../core/config/verify.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The chain of segment `segment` read off its rows as they stand, from its
/// first end to its last: `f(a, b, n0, n1)` per piece, its two sites in the
/// segment's direction and their numerators over 2^depth, `-1` for a site
/// off the frame, `depth` at least the segment's frame bits. The rows are
/// the chain's one producer, so a reader sees every split, join and rename
/// the moment it is stated. O(the chain).
template <typename Index, typename Coord, typename F>
auto for_each_tet_chain_piece(
    const tet_segment_recovery<Index, Coord> &recovery, std::size_t segment,
    int depth, const F &f) -> void {
  const auto &rows = recovery.subsegments;
  const int lift = depth - recovery.bits[segment];
  if constexpr (tf::core::verification)
    tf::core::verify(lift >= 0, "a chain is read at its frame or finer");
  const auto lifted = [lift](std::int64_t numerator) {
    return numerator < 0 ? numerator : numerator << lift;
  };
  for (Index k = recovery.first[segment]; k != Index(-1);
       k = rows[std::size_t(k)].next) {
    const auto &row = rows[std::size_t(k)];
    f(row.a, row.b, lifted(row.n0), lifted(row.n1));
  }
}

} // namespace tf::topology::cdt::dt3
