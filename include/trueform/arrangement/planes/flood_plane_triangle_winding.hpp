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
#include "../../topology/winding/flood_triangle_winding.hpp"
#include "../../topology/winding/loop_touches_itself.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace tf::arrangement {

/// Whether a lone member's loop touches itself
/// (@ref tf::topology::loop_touches_itself), its passes read off the plane's
/// constraints and tallied on the triangulation's vertices.
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
  return tf::topology::loop_touches_itself(
      cons_winding.size(),
      [&](std::size_t constraint) {
        const auto winding = Index(std::abs(int(cons_winding[constraint])));
        const auto multiplicity =
            winding != Index(0) || cons_side[constraint] < short(0) ? winding
                                                                    : Index(2);
        return std::array<Index, 3>{
            Index(forward[std::size_t(cons[constraint * 2])]),
            Index(forward[std::size_t(cons[constraint * 2 + 1])]),
            multiplicity};
      },
      [&degree](Index vertex, Index passes) {
        return degree[std::size_t(vertex)] += passes;
      });
}

/// The winding of a lone member's loop around every triangle of its carrier
/// (@ref tf::topology::flood_triangle_winding), read off its triangulation.
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
  tf::topology::flood_triangle_winding(
      n_triangles,
      [&neighbors](Index triangle, int edge) {
        return neighbors[std::size_t(triangle)][std::size_t(edge)];
      },
      entering, stack, winding);
}

} // namespace tf::arrangement
