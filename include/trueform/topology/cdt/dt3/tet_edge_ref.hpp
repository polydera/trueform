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
#include "../../../core/config/verify.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// An edge of a cell, read as the ordered corners [a, b, c, d]: the cell
/// and the slots of a and b. A finite cell's stored corners are positively
/// oriented and every such reading is an even permutation of them, so the
/// apex c and the oppo d follow from the two slots by parity alone
/// (@ref tet_edge_apex_slot), and every other reading of the cell is a
/// choice of two slots: [b, c, a, d] is @ref tet_edge_turn, [c, a, b, d]
/// @ref tet_edge_turn_back, [b, a, d, c] @ref tet_edge_sym, [c, b, d, a]
/// @ref tet_edge_turn_sym, [a, c, d, b] @ref tet_edge_turn_back_sym,
/// [c, d, a, b] @ref tet_edge_apex_oppo, [d, c, b, a]
/// @ref tet_edge_oppo_apex, [b, a, c, e] across the facet a b c
/// @ref tet_edge_across, and [a, b, d, x], the next cell about a b in
/// right-hand order about a -> b, @ref tet_edge_next. A hull cell is an
/// edge's cell like any other, its `infinite` corner the apex or the oppo.
template <typename Index> struct tet_edge_ref {
  Index cell;
  std::uint8_t a;
  std::uint8_t b;
};

/// The apex slot of the edge at slots `[a][b]`; the oppo slot is the
/// remaining one.
inline constexpr std::uint8_t tet_edge_apex_slot[4][4] = {
    {255, 2, 3, 1}, {3, 255, 0, 2}, {1, 3, 255, 0}, {2, 0, 1, 255}};

template <typename Index>
auto tet_edge_apex(const tet_edge_ref<Index> &edge) -> std::uint8_t {
  return tet_edge_apex_slot[edge.a][edge.b];
}

template <typename Index>
auto tet_edge_oppo(const tet_edge_ref<Index> &edge) -> std::uint8_t {
  return std::uint8_t(6 - edge.a - edge.b - tet_edge_apex(edge));
}

template <typename Index>
auto tet_edge_sym(const tet_edge_ref<Index> &edge) -> tet_edge_ref<Index> {
  return {edge.cell, edge.b, edge.a};
}

template <typename Index>
auto tet_edge_turn(const tet_edge_ref<Index> &edge) -> tet_edge_ref<Index> {
  return {edge.cell, edge.b, tet_edge_apex(edge)};
}

template <typename Index>
auto tet_edge_turn_back(const tet_edge_ref<Index> &edge)
    -> tet_edge_ref<Index> {
  return {edge.cell, tet_edge_apex(edge), edge.a};
}

template <typename Index>
auto tet_edge_turn_sym(const tet_edge_ref<Index> &edge)
    -> tet_edge_ref<Index> {
  return {edge.cell, tet_edge_apex(edge), edge.b};
}

template <typename Index>
auto tet_edge_turn_back_sym(const tet_edge_ref<Index> &edge)
    -> tet_edge_ref<Index> {
  return {edge.cell, edge.a, tet_edge_apex(edge)};
}

template <typename Index>
auto tet_edge_apex_oppo(const tet_edge_ref<Index> &edge)
    -> tet_edge_ref<Index> {
  return {edge.cell, tet_edge_apex(edge), tet_edge_oppo(edge)};
}

template <typename Index>
auto tet_edge_oppo_apex(const tet_edge_ref<Index> &edge)
    -> tet_edge_ref<Index> {
  return {edge.cell, tet_edge_oppo(edge), tet_edge_apex(edge)};
}

/// The edge from `a` to `b` in `cell`, both corners of it.
template <typename Owner>
auto tet_edge_of(const Owner &owner, typename Owner::index_type cell,
                 typename Owner::index_type a, typename Owner::index_type b)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto corners = owner._corners[std::size_t(cell)];
  std::uint8_t sa = 0, sb = 0;
  while (sa < 3 && corners[sa] != a)
    ++sa;
  while (sb < 3 && corners[sb] != b)
    ++sb;
  if constexpr (tf::core::verification)
    tf::core::verify(corners[sa] == a && corners[sb] == b && sa != sb,
                     "the cell holds both corners");
  return {cell, sa, sb};
}

/// The corners [a, b, c, d] of the edge's cell in the edge's reading.
template <typename Owner>
auto tet_edge_corners(const Owner &owner,
                      const tet_edge_ref<typename Owner::index_type> &edge)
    -> std::array<typename Owner::index_type, 4> {
  const auto corners = owner._corners[std::size_t(edge.cell)];
  return {corners[edge.a], corners[edge.b], corners[tet_edge_apex(edge)],
          corners[tet_edge_oppo(edge)]};
}

template <typename Owner>
auto tet_edge_next(const Owner &owner,
                   const tet_edge_ref<typename Owner::index_type> &edge)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto corners = owner._corners[std::size_t(edge.cell)];
  const auto next =
      owner._neighbors[std::size_t(edge.cell)][tet_edge_apex(edge)];
  return tet_edge_of(owner, next, corners[edge.a], corners[edge.b]);
}

template <typename Owner>
auto tet_edge_across(const Owner &owner,
                     const tet_edge_ref<typename Owner::index_type> &edge)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto corners = owner._corners[std::size_t(edge.cell)];
  const auto other =
      owner._neighbors[std::size_t(edge.cell)][tet_edge_oppo(edge)];
  return tet_edge_of(owner, other, corners[edge.b], corners[edge.a]);
}

} // namespace tf::topology::cdt::dt3
