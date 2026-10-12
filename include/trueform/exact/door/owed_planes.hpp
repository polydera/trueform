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

#include "../canonical_plane.hpp"
#include "./placement_tables.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::exact::door {

/// The most helper planes one vertex weighs. The cut is made after the
/// sort, so two vertices carrying the same planes still see the same list;
/// it bounds the widest-triple search, which is cubic in the list. Owed
/// planes are never cut.
inline constexpr std::size_t helper_limit = 24;

/// The planes a vertex owes, and the planes that may only choose where it
/// lands.
///
/// It owes the committed plane of every face it belongs to and of every
/// face it was admitted against, and every identification it keeps: the
/// plane of a face it stands on exactly — the committed one when the face
/// committed — and the plane of every exact edge contact it spans. Faces of
/// `skip_group` are the group being decided: they owe the candidate the
/// caller adds, and nothing here. With `contacts` false the faces it was
/// admitted against bind it only through what it keeps, which is what a
/// vertex falls back to when the new pulls cannot all hold it.
///
/// The helpers are the exact planes of its own uncommitted faces: they pass
/// through the vertex, so a meet with them slides it along its own surface.
template <typename Index, typename Int, typename RealType, typename PlaneOf,
          typename GroupOf, typename Owed, typename Helpers>
auto gather_owed_planes(
    const placement_tables<Index, Int, RealType> &tables,
    const PlaneOf &plane_of_face,
    const tf::buffer<tf::exact::canonical_plane<Int>> &plane,
    const GroupOf &group_of_face, Index skip_group, std::size_t vertex,
    bool contacts, Owed &owed, Helpers &helpers) -> void {
  const tf::exact::canonical_plane<Int> none{};
  owed.clear();
  helpers.clear();
  for (auto k = tables.incidence_offsets[vertex];
       k < tables.incidence_offsets[vertex + 1]; ++k) {
    const auto f = std::size_t(tables.incidence[std::size_t(k)]);
    if (group_of_face(f) == skip_group)
      continue;
    if (plane_of_face[f] >= 0)
      owed.push_back(plane[std::size_t(plane_of_face[f])]);
    else if (tables.planes[f] != none)
      helpers.push_back(tables.planes[f]);
  }
  for (auto k = tables.contact_offsets[vertex];
       k < tables.contact_offsets[vertex + 1]; ++k) {
    const auto f = std::size_t(tables.contacts[std::size_t(k)]);
    if (group_of_face(f) == skip_group)
      continue;
    if (contacts && plane_of_face[f] >= 0)
      owed.push_back(plane[std::size_t(plane_of_face[f])]);
  }
  if (tables.held_offsets.size() != 0)
    for (auto k = tables.held_offsets[vertex];
         k < tables.held_offsets[vertex + 1]; ++k) {
      const auto row = tables.held[std::size_t(k)];
      if (row < Index(0)) {
        owed.push_back(tables.held_planes[std::size_t(-1 - row)]);
        continue;
      }
      const auto f = std::size_t(row);
      if (group_of_face(f) == skip_group)
        continue;
      if (plane_of_face[f] >= 0)
        owed.push_back(plane[std::size_t(plane_of_face[f])]);
      else if (tables.planes[f] != none)
        owed.push_back(tables.planes[f]);
    }
  std::sort(owed.begin(), owed.end());
  owed.erase(std::unique(owed.begin(), owed.end()), owed.end());
  std::sort(helpers.begin(), helpers.end());
  helpers.erase(std::unique(helpers.begin(), helpers.end()), helpers.end());
  if (helpers.size() > helper_limit)
    helpers.resize(helper_limit);
}

} // namespace tf::exact::door
