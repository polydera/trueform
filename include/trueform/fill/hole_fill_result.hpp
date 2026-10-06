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
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/point.hpp"
#include "./hole_fill_status.hpp"
#include "./hole_split.hpp"
#include <array>
#include <cstddef>

namespace tf {

/// @ingroup fill
/// @brief What @ref tf::fill_holes made of a mesh's rims.
///
/// The carrier is the group: block `g` of every jagged lane and slot `g` of
/// every flat lane belong to group `g`, which `group_offsets` states as a
/// run of rims — one rim per group here, the outer rim of a group first.
///
/// A corner is a flat identity: below the input mesh's point count it is one
/// of its vertices, and at or above it names `minted_points` read flat, in
/// group order. Group `g`'s own mints are block `g` of that lane, so the
/// stitched read appends the whole lane once, after the input's points.
///
/// A refused group publishes nothing — an empty triangle block, an empty
/// mint block, no splits and no plan participation — and names the rim edge
/// it refused on. `offending` is `(rim, edge)`, `(-1, -1)` where there is
/// none.
///
/// `splits` is the canonical split table: the original edges the fill cut,
/// each with the point it resolved to. `plan_faces` and `plan_triangles`
/// state what every carrying face those splits reached becomes — the one
/// authority the stitched read replays.
///
/// @tparam Index The index type of the mesh the result is stated against.
/// @tparam RealT The coordinate type minted points are carried in.
template <typename Index, typename RealT> struct hole_fill_result {
  /// Per group, the run of rims it fills.
  tf::buffer<Index> group_offsets;
  /// Per group, the patch it holds.
  tf::offset_block_buffer<Index, std::array<Index, 3>> triangles;
  /// Per group, the points it minted.
  tf::offset_block_buffer<Index, tf::point<RealT, 3>> minted_points;
  /// Per group, what its fill produced.
  tf::buffer<tf::hole_fill_status> status;
  /// Per group, how its patch left refinement.
  tf::buffer<tf::hole_refine_status> refined;
  /// Per group, how its patch left fairing.
  tf::buffer<tf::hole_fair_status> faired;
  /// Per group, the largest angle across its seam, `-1` when not computed.
  tf::buffer<RealT> seam_max_angle;
  /// Per group, the `(rim, edge)` its status speaks of.
  tf::blocked_buffer<Index, 2> offending;
  /// The original edges the surviving groups cut, in canonical order.
  tf::buffer<tf::hole_split<Index>> splits;
  /// The carrying faces those splits reached, ascending.
  tf::buffer<Index> plan_faces;
  /// Per face of `plan_faces`, the triangles it becomes.
  tf::offset_block_buffer<Index, std::array<Index, 3>> plan_triangles;

  /// The number of groups.
  auto size() const -> std::size_t { return status.size(); }
};

} // namespace tf
