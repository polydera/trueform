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
#include "./flip_tet_ring.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_flip_constraints.hpp"
#include "./tet_flip_scratch.hpp"
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Remove the edge of `edge` by flips: its
/// ring is gathered into the arena and flipped by @ref flip_tet_ring at
/// level 0; what the ring's flips made is walked back by
/// @ref unflip_tet_ring without undoing. A segment is not removed, and is
/// noted when `fc` asks; a ring past the star limit is left alone. Returns
/// 2 when the edge is removed, 0 when it was not tried, and the ring's
/// size otherwise, `edge` then its first cell.
template <typename Owner, typename IsSegment>
auto remove_tet_edge_by_flips(
    Owner &owner, tet_flip_scratch<typename Owner::index_type> &scratch,
    tet_edge_ref<typename Owner::index_type> &edge,
    const tet_flip_constraints<typename Owner::index_type,
                               typename Owner::coord_type,
                               typename Owner::int_type> &fc,
    const IsSegment &is_segment) -> int {
  using Index = typename Owner::index_type;
  using entry_t = tet_ring_entry<Index>;
  auto &arena = scratch.arena;
  const auto c = tet_edge_corners(owner, edge);
  if (is_segment(c[0], c[1])) {
    if (fc.collect_encroached)
      note_tet_encroached_segment(scratch, c[0], c[1]);
    return 0;
  }
  int n = 0;
  auto spin = edge;
  do {
    ++n;
    spin = tet_edge_next(owner, spin);
  } while (spin.cell != edge.cell);
  assert(n >= 3);
  if (scratch.star_limit > 0 && n > scratch.star_limit)
    return 0;
  const std::size_t begin = arena.size();
  spin = edge;
  for (int i = 0; i < n; ++i) {
    auto &e = arena.push_back(entry_t{});
    e.cell = spin.cell;
    e.a = spin.a;
    e.b = spin.b;
    tet_ring_count(scratch, spin.cell) = 1;
    spin = tet_edge_next(owner, spin);
  }
  const int nn = flip_tet_ring(owner, scratch, begin, n, 0, 0, fc, is_segment);
  if (nn > 2) {
    for (int i = 0; i < nn; ++i)
      tet_ring_count(scratch, arena[begin + std::size_t(i)].cell) = 0;
    const auto &e = arena[begin];
    edge = {e.cell, e.a, e.b};
  }
  auto walk = fc;
  walk.unflip = false;
  unflip_tet_ring(owner, scratch, begin, n, nn, 0, walk, is_segment);
  arena.erase_till_end(arena.begin() + std::ptrdiff_t(begin));
  return nn;
}

} // namespace tf::topology::cdt::dt3
