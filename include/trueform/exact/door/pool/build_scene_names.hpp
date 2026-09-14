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

#include "./build_name_registry.hpp"
#include "./elect_support_points.hpp"
#include "./find_scene_features.hpp"
#include "./gather_name_supports.hpp"
#include "./name_face_planes.hpp"
#include "./pool_records.hpp"
#include "./scene_targets.hpp"
#include "./state_name_witnesses.hpp"

#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../canonical_plane.hpp"
#include "../placement_tables.hpp"

namespace tf::exact::door::pool {

/// The world's original name space, as the partition consumes it.
///
/// A NAME IS AN EXACT SUPPORT PLANE — the canonical quadruple through the
/// three original lattice vertices a face stands on. Nothing quantized
/// enters the name space: the grid the door rounds a plane onto has a
/// different equality from the plane itself.
///
/// THE ONE SORT IS THE WHOLE PASS. Faces sorted by their own exact plane
/// give the distinct names, the name every face speaks, and the block of
/// faces each name owns — and that block is the grain the support point and
/// the support set are then read at, so no later phase searches for a name.
template <typename Index, typename Int, typename RealType>
auto build_scene_names(const placement_tables<Index, Int, RealType> &tables,
                       const tf::blocked_buffer<Index, 3> &corners,
                       Int tolerance, pool_names<Int> &names,
                       scene_targets &targets) -> void {
  tf::buffer<tf::exact::canonical_plane<Int>> face_plane;
  name_face_planes(tables, corners, face_plane);

  tf::buffer<int> face_name;
  tf::offset_block_buffer<int, Index> name_faces;
  build_name_registry<Index, Int>(face_plane, names.plane, face_name,
                                  name_faces);

  find_scene_features(tables, face_plane, face_name, tolerance, targets);

  elect_support_points(tables, corners, name_faces, names.support_point);
  gather_name_supports(tables, corners, name_faces, targets.feature,
                       names.support_offsets, names.support);
  state_name_witnesses(tables, corners, face_name, targets, names.witness);
}

} // namespace tf::exact::door::pool
