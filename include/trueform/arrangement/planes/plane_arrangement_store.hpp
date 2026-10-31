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
#include "../../core/offset_block_buffer.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../topology/topo_id.hpp"
#include "./plane_triangulation_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// The published product, plane-major: the triangles and every table that
/// rides them, each plane's span and each face's, and the definition span of
/// every piece the plane tier owns. The arena is where a build produces; this
/// is what it publishes.
template <typename Index> struct plane_arrangement_store {
  using coplanar_t = tf::arrangement::coplanar_descriptor<Index>;
  using def_t = tf::intersect::graph::plane_edge_def<Index>;

  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<std::array<Index, 3>> slot_parents;
  tf::buffer<std::array<tf::topo_id<short>, 3>> corner_subs;
  tf::buffer<Index> plane_offsets;
  tf::buffer<std::array<Index, 2>> face_range;
  tf::buffer<Index> coplanar_of;
  tf::buffer<coplanar_t> coplanar;
  tf::buffer<char> stacked;
  tf::buffer<Index> triangle_cells;
  tf::offset_block_buffer<Index, def_t> final_piece_definitions;

  /// No triangle and no table that rides one.
  auto clear_triangles() -> void {
    triangles.clear();
    slot_parents.clear();
    corner_subs.clear();
    coplanar_of.clear();
    coplanar.clear();
    stacked.clear();
    triangle_cells.clear();
    final_piece_definitions.clear();
  }
  /// The store holds nothing, not even a plane.
  auto clear() -> void {
    clear_triangles();
    plane_offsets.clear();
    face_range.clear();
  }
  /// Every plane and every face of a world holds an empty span.
  auto reset(Index n_planes, Index n_faces) -> void {
    clear_triangles();
    plane_offsets.allocate(std::size_t(n_planes) + 1);
    std::fill(plane_offsets.begin(), plane_offsets.end(), Index(0));
    face_range.allocate(std::size_t(n_faces));
    std::fill(face_range.begin(), face_range.end(),
              std::array<Index, 2>{Index(0), Index(0)});
  }
};

} // namespace tf::arrangement
