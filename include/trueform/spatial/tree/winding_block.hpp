/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/range.hpp"
#include <array>
#include <cstdint>

namespace tf::spatial {

/// One child block of second-order winding expansions,
/// attribute-major: every stored quantity holds four children's values
/// side by side, so the far test and the far-field polynomial run
/// lane-parallel over the whole block. An inner node owns one block
/// per four children, consecutive from its ticket, each naming its own
/// child sub-range. Lanes are in child order; an absent child is a
/// zero lane, and a zero area lane states its own irrelevance. The quantities
/// are the far-field evaluation's own combinations: centroid, area, dipole,
/// squared far-field radius, the symmetric second-moment parts (diagonal; cross
/// sums Nxy+Nyx, Nzx+Nxz, Nyz+Nzy), and the third tier's ten (diagonal Niii,
/// the xyz permutation sum, the six 2Niij+Njii).
template <typename RealT> struct winding_block {
  using lane = std::array<RealT, 4>;
  lane position[3];
  lane area;
  lane directed_area[3];
  lane far_field_radius2;
  lane second_diag[3];
  lane second_cross[3];
  lane third_diag[3];
  lane third_permute;
  lane third_mixed[6];
  std::int32_t first_child;
  std::int32_t n_children;
};

static_assert(sizeof(winding_block<float>) == 392);

/// The sub-blocks an inner node owns for its children — one per four,
/// consecutive from its ticket.
inline auto sub_block_count(std::size_t n_children) -> std::size_t {
  return (n_children + 3) / 4;
}

/// The winding moments as a consumer holds them: the child blocks and
/// the per-node ticket of the FIRST block an inner node owns (`-1`
/// elsewhere). A ticket row exists for every tree node, so the view's
/// size is the node count it was built over.
template <typename RealT> struct winding_moments_view {
  tf::range<const winding_block<RealT> *, tf::dynamic_size> blocks;
  tf::range<const std::int32_t *, tf::dynamic_size> tickets;

  auto size() const { return tickets.size(); }
};

} // namespace tf::spatial
