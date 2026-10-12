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
#include "../../core/point.hpp"
#include "../../exact/edge_parameter.hpp"
#include "../../exact/parameter_line_sign2.hpp"
#include "./constrained_delaunay_edge_constraint_provenance.hpp"
#include "./is_between_constrained_delaunay_vertices.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// A constrained edge, from `first` to `second`, that a meeting lands on.
template <typename Index, typename Provenance>
struct constrained_delaunay_meeting_landing {
  Index edge;
  Index first;
  Index second;
  bool is_boundary;
  Provenance provenance;
};

/// The constrained edges a meeting's rounding reached. The exact crossing of
/// (`first`, `second`) at `parameter` lies on the constrained edge `crossed`;
/// its lattice point `meeting` stands within rounding of it. Every
/// constrained edge the segment from the crossing to `meeting` meets, with
/// `meeting` on it, is a constraint that point lies on at lattice resolution
/// and is pushed to `landings`, walking the faces that segment passes
/// through. A segment through a vertex ends the walk.
template <typename Owner, typename Landings>
auto find_constrained_delaunay_meeting_landings(
    const Owner &owner, typename Owner::index_type first,
    typename Owner::index_type second,
    const tf::exact::edge_parameter<typename Owner::int_type> &parameter,
    typename Owner::index_type crossed,
    const tf::point<typename Owner::int_type, 2> &meeting, Landings &landings)
    -> void {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  const auto from = owner.point(first);
  const auto to = owner.point(second);
  const auto turn_of = [&](Index vertex) {
    return tf::exact::parameter_line_sign2<Int>(meeting, owner.point(vertex),
                                                from, to, parameter);
  };
  const auto land = [&](Index edge) {
    if (owner._edges[std::size_t(edge)].constrained == Owner::unconstrained)
      return;
    landings.push_back(
        {edge, owner.origin(edge), owner.target(edge),
         owner._edges[std::size_t(edge)].constrained ==
             Owner::boundary_constrained,
         constrained_delaunay_edge_constraint_provenance(owner, edge)});
  };

  const int side =
      owner.orient(owner.origin(crossed), owner.target(crossed), meeting);
  if (side == 0)
    return;
  Index entry = side < 0 ? crossed : Owner::opposite(crossed);
  for (std::size_t guard = 0; guard < owner._edges.size(); ++guard) {
    if (owner._edges[std::size_t(entry)].boundary)
      return;
    const Index second_edge = owner.previous_edge(Owner::opposite(entry));
    const Index third_edge = owner.previous_edge(Owner::opposite(second_edge));
    const int second_sign = owner.orient(owner.origin(second_edge),
                                         owner.target(second_edge), meeting);
    const int third_sign = owner.orient(owner.origin(third_edge),
                                        owner.target(third_edge), meeting);
    Index exit = Owner::none;
    if (second_sign > 0 && third_sign > 0) {
      const int apex_turn = turn_of(owner.origin(third_edge));
      if (apex_turn == 0)
        return;
      exit = apex_turn == turn_of(owner.origin(second_edge)) ? third_edge
                                                             : second_edge;
    } else if (second_sign > 0) {
      exit = second_edge;
    } else if (third_sign > 0) {
      exit = third_edge;
    } else {
      if (second_sign == 0 &&
          is_between_constrained_delaunay_vertices(
              owner, owner.origin(second_edge), owner.target(second_edge),
              meeting))
        land(second_edge);
      if (third_sign == 0 &&
          is_between_constrained_delaunay_vertices(
              owner, owner.origin(third_edge), owner.target(third_edge),
              meeting))
        land(third_edge);
      return;
    }
    land(exit);
    entry = Owner::opposite(exit);
  }
}

} // namespace tf::topology::cdt
