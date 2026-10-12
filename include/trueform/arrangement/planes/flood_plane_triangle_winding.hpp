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
#include "../../exact/meta.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace tf::arrangement {

/// Whether a lone member's loop touches itself: some vertex of the
/// triangulation carries more than two of its passes. A loop that does not is
/// a simple cycle, and its winding is its parity everywhere.
///
/// A wall the loop states both ways cancels its winding, not its passes: the
/// loop still runs it twice, and what it joins is still one loop. A wall no
/// side states (`cons_side` negative) is a cut and passes nothing.
///
/// `forward` maps a prepared point to the triangulation's vertex, so two
/// identities the projection welded meet at one vertex here.
template <typename Index, typename Forward>
auto plane_loop_touches_itself(const tf::buffer<Index> &cons,
                               const tf::buffer<short> &cons_side,
                               const tf::buffer<signed char> &cons_winding,
                               const Forward &forward, std::size_t n_vertices,
                               tf::buffer<Index> &degree) -> bool {
  degree.allocate(n_vertices);
  std::fill(degree.begin(), degree.end(), Index(0));
  for (std::size_t constraint = 0; constraint < cons_winding.size();
       ++constraint) {
    const auto winding = Index(std::abs(int(cons_winding[constraint])));
    const auto multiplicity =
        winding != Index(0) || cons_side[constraint] < short(0) ? winding
                                                                : Index(2);
    if (multiplicity == Index(0))
      continue;
    for (int end = 0; end < 2; ++end) {
      auto &count = degree[std::size_t(
          forward[std::size_t(cons[constraint * 2 + std::size_t(end)])])];
      count += multiplicity;
      if (count > Index(2))
        return true;
    }
  }
  return false;
}

/// The winding of a lone member's loop around every triangle of its carrier.
///
/// The hull exterior winds nothing, and a wall moves the winding by the
/// multiplicity the member states it with, signed by the side it is crossed
/// from: a triangle running the wall's own direction stands left of it in
/// the triangulation's own winding `cdt_orientation`. `cons_root` names the
/// constraint an aliased owner speaks for, empty when every owner speaks for
/// itself. A root's winding is stated in the root's own direction, and an
/// alias may run against it, so the owner's direction is turned into the
/// root's by the sign of their two runs on the prepared points `pts2`.
template <typename Index, typename Cdt, typename Points2>
auto flood_plane_triangle_winding(const Cdt &cdt, int cdt_orientation,
                                  const tf::buffer<Index> &cons,
                                  const Points2 &pts2,
                                  const tf::buffer<signed char> &cons_winding,
                                  const tf::buffer<Index> &cons_root,
                                  tf::buffer<std::array<Index, 3>> &neighbors,
                                  tf::buffer<Index> &stack,
                                  tf::buffer<signed char> &winding) -> void {
  using Int = std::decay_t<decltype(pts2[0][0])>;
  using T2 = typename tf::exact::meta<Int>::T2;
  // +1 where constraints `a` and `b` run the same way, -1 where they oppose
  const auto runs_with = [&](Index a, Index b) {
    const auto &a0 = pts2[std::size_t(cons[2 * std::size_t(a)])];
    const auto &a1 = pts2[std::size_t(cons[2 * std::size_t(a) + 1])];
    const auto &b0 = pts2[std::size_t(cons[2 * std::size_t(b)])];
    const auto &b1 = pts2[std::size_t(cons[2 * std::size_t(b) + 1])];
    const T2 dot = (T2(a1[0]) - T2(a0[0])) * (T2(b1[0]) - T2(b0[0])) +
                   (T2(a1[1]) - T2(a0[1])) * (T2(b1[1]) - T2(b0[1]));
    return dot < T2(0) ? -1 : 1;
  };
  const auto n_triangles = std::size_t(cdt.region_labels().size());
  neighbors.allocate(n_triangles);
  cdt.for_each_face_adjacency([&](Index triangle, Index, Index, Index, Index,
                                  const std::array<Index, 3> &across,
                                  const std::array<bool, 3> &) {
    neighbors[std::size_t(triangle)] = across;
  });
  // the winding a wall adds entering `triangle` across its edge `edge`
  const auto entering = [&](Index triangle, int edge) {
    const auto owner = cdt.face_constraint_owners(triangle)[std::size_t(edge)];
    if (owner.input_id == Index(-1))
      return 0;
    const auto input = cons_root.size() == 0
                           ? owner.input_id
                           : cons_root[std::size_t(owner.input_id)];
    int along = owner.t0 < owner.t1 ? 1 : -1;
    if (input != owner.input_id)
      along *= runs_with(owner.input_id, input);
    return int(cons_winding[std::size_t(input)]) * along * cdt_orientation;
  };
  const auto unset = static_cast<signed char>(SCHAR_MIN);
  winding.allocate(n_triangles);
  std::fill(winding.begin(), winding.end(), unset);
  stack.clear();
  for (std::size_t triangle = 0; triangle < n_triangles; ++triangle)
    for (int edge = 0; edge < 3; ++edge)
      if (neighbors[triangle][std::size_t(edge)] == Index(-1)) {
        winding[triangle] =
            static_cast<signed char>(entering(Index(triangle), edge));
        stack.push_back(Index(triangle));
        break;
      }
  while (stack.size() != 0) {
    const auto triangle = stack[stack.size() - 1];
    stack.pop_back();
    for (int edge = 0; edge < 3; ++edge) {
      const auto next = neighbors[std::size_t(triangle)][std::size_t(edge)];
      if (next == Index(-1) || winding[std::size_t(next)] != unset)
        continue;
      winding[std::size_t(next)] = static_cast<signed char>(
          winding[std::size_t(triangle)] - entering(triangle, edge));
      stack.push_back(next);
    }
  }
}

} // namespace tf::arrangement
