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
#include "./pool_records.hpp"
#include "./scene_targets.hpp"
#include "./state_name_witnesses.hpp"

#include "../../../core/buffer.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../canonical_plane.hpp"
#include "../placement_tables.hpp"

namespace tf::exact::door::pool {

/// The world's original name space, as the partition consumes it.
///
/// A NAME IS AN EXACT SUPPORT PLANE — the canonical quadruple through the
/// three original lattice vertices a face stands on, which the tables
/// already state for every face.
///
/// THE ONE SORT IS THE WHOLE PASS. Faces sorted by their own exact plane
/// give the distinct names, the name every face speaks, and the block of
/// faces each name owns — and that block is the grain the support point and
/// the support set are then read at, so no later phase searches for a name.
///
/// The exact plane every face stands on is the tables' own, so the name a
/// face speaks here and the plane a contact against it lands on are one
/// quadruple. `face_name` is that name's id per flat face, `-1` for a face
/// that names no plane.
template <typename Index, typename Int, typename RealType>
auto build_scene_names(const placement_tables<Index, Int, RealType> &tables,
                       Int tolerance, pool_names<Int> &names,
                       scene_targets &targets, tf::buffer<int> &face_name)
    -> void {
  tf::offset_block_buffer<int, Index> name_faces;
  build_name_registry<Index, Int>(tables.planes, names.plane, face_name,
                                  name_faces);

  find_scene_features(tables, tables.planes, face_name, tolerance, targets);

  elect_support_points(tables, name_faces, names.support_point);
  gather_name_supports(tables, name_faces, targets.feature,
                       names.support_offsets, names.support);
  state_name_witnesses(tables, face_name, targets, names.witness);
}

} // namespace tf::exact::door::pool
