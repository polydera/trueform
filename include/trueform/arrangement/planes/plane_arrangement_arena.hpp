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
#include "../../exact/vertex.hpp"
#include "../../topology/topo_id.hpp"
#include "./plane_triangulation_types.hpp"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// The per-plane blocks as they are produced, with each plane's ranges
/// into them. Slot values are provisional constraint-group markers; the
/// output boundary states them from the definition table. Task-local link
/// tickets translate into this arena's rows exactly once after aggregation.
template <typename Index, typename Int> struct plane_arrangement_arena {
  using pt3_t = tf::exact::pt3<Int>;
  using coplanar_t = tf::arrangement::coplanar_descriptor<Index>;

  tf::buffer<std::array<Index, 3>> tris;
  tf::buffer<std::array<Index, 3>> slots;
  tf::buffer<std::array<tf::topo_id<short>, 3>> subs;
  tf::buffer<Index> coplanar_of;
  tf::buffer<char> stacked;
  tf::buffer<coplanar_t> coplanar;
  tf::buffer<pt3_t> steiners;
  tf::buffer<Index> cell_of;                   // per triangle, requested
  tf::buffer<std::array<Index, 2>> range;      // per plane, into tris
  tf::buffer<std::array<Index, 2>> cop_range;  // per plane, into coplanar
  tf::buffer<std::array<Index, 2>> stn_range;  // per plane, into steiners
  tf::buffer<std::array<Index, 2>> face_range; // per face, into tris

  auto clear() -> void {
    tris.clear();
    slots.clear();
    subs.clear();
    coplanar_of.clear();
    stacked.clear();
    coplanar.clear();
    steiners.clear();
    cell_of.clear();
    range.clear();
    cop_range.clear();
    stn_range.clear();
    face_range.clear();
  }
  auto allocate_planes(Index n) -> void {
    const auto empty = std::array<Index, 2>{Index(0), Index(0)};
    range.allocate_and_initialize(std::size_t(n), empty);
    cop_range.allocate_and_initialize(std::size_t(n), empty);
    stn_range.allocate_and_initialize(std::size_t(n), empty);
  }
  /// The preserving sibling of @ref allocate_planes: a carrier the arena
  /// already holds keeps the span it was given, and the carriers a promotion
  /// appended start empty like every other.
  auto grow_planes(Index n) -> void {
    const auto empty = std::array<Index, 2>{Index(0), Index(0)};
    range.reallocate_and_initialize(std::size_t(n), empty);
    cop_range.reallocate_and_initialize(std::size_t(n), empty);
    stn_range.reallocate_and_initialize(std::size_t(n), empty);
  }
  auto grow_faces(Index n) -> void {
    face_range.reallocate_and_initialize(std::size_t(n),
                                         {Index(0), Index(0)});
  }
};

} // namespace tf::arrangement
