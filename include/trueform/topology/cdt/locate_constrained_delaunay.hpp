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
#include "./find_constrained_delaunay_interior_edge.hpp"
#include "./is_between_constrained_delaunay_vertices.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Walk from the retained hint to the face containing `query`. The face cycle
/// is clockwise, so the query belongs to a face when it is not strictly left of
/// any face edge. After crossing an edge it is already known to lie to that
/// edge's right, leaving at most two new orientation tests per face.
template <typename Owner, typename Query>
auto locate_constrained_delaunay(Owner &owner, const Query &query) ->
    typename Owner::locate_result {
  using Index = typename Owner::index_type;
  using LocateKind = typename Owner::locate_kind;
  Index first = owner._locate_hint;
  if (first == Owner::none || owner._edges[std::size_t(first)].boundary)
    first = find_constrained_delaunay_interior_edge(owner);

  int first_sign =
      owner.orient(owner.origin(first), owner.target(first), query);
  if (first_sign > 0) {
    if (owner._edges[std::size_t(Owner::opposite(first))].boundary)
      return {LocateKind::exterior, Owner::opposite(first)};
    first = Owner::opposite(first);
    first_sign = -1;
  }

  const Index n_darts = static_cast<Index>(owner._edges.size());
  for (Index guard = 0; guard <= n_darts; ++guard) {
    const Index second = owner.previous_edge(Owner::opposite(first));
    const Index third = owner.previous_edge(Owner::opposite(second));

    const int second_sign =
        owner.orient(owner.origin(second), owner.target(second), query);
    if (second_sign > 0) {
      if (owner._edges[std::size_t(Owner::opposite(second))].boundary)
        return {LocateKind::exterior, Owner::opposite(second)};
      first = Owner::opposite(second);
      first_sign = -1;
      continue;
    }

    const int third_sign =
        owner.orient(owner.origin(third), owner.target(third), query);
    if (third_sign > 0) {
      if (owner._edges[std::size_t(Owner::opposite(third))].boundary)
        return {LocateKind::exterior, Owner::opposite(third)};
      first = Owner::opposite(third);
      first_sign = -1;
      continue;
    }

    if (first_sign == 0 || second_sign == 0 || third_sign == 0) {
      const Index edges[3] = {first, second, third};
      const int signs[3] = {first_sign, second_sign, third_sign};
      for (int i = 0; i < 3; ++i) {
        if (signs[i] != 0 ||
            !is_between_constrained_delaunay_vertices(
                owner, owner.origin(edges[i]), owner.target(edges[i]), query))
          continue;
        if (owner._edges[std::size_t(Owner::opposite(edges[i]))].boundary)
          return {LocateKind::boundary_edge, Owner::opposite(edges[i])};
        return {LocateKind::interior, first};
      }
    }
    return {LocateKind::interior, first};
  }
  // The guard bounds a walk over malformed topology without adding a second
  // failure state to the locator's face-or-boundary result.
  return {LocateKind::interior, first};
}

} // namespace tf::topology::cdt
