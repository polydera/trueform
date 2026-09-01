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
#include "./buffer.hpp"
#include "./offset_block_buffer.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief How several input meshes became one output, for carrying their
///        structures across instead of rebuilding them.
///
/// A face that survived whole keeps whatever an input structure recorded about
/// it — its peers, its vertices' incidences, its bounding volume — so only its
/// id is remapped. A face with no input counterpart has to be rebuilt from the
/// result. That split, plus the two id maps, is everything a stitched structure
/// needs.
///
/// Ids here are the output's own, never ordinals within a source plus an
/// offset. Nothing in this map implies an emission order, so it holds for any
/// producer that merges meshes.
///
/// @tparam Index The index type.
template <typename Index> struct stitch_index_map {
  /// face_f[source][input face] -> output face; no counterpart -> n_output_faces
  tf::offset_block_buffer<Index, Index> face_f;
  /// output point -> the source it came from; a new point -> n_sources
  tf::buffer<Index> point_sources;
  /// output point -> its point id within that source
  tf::buffer<Index> point_ids;
  /// output faces with no input counterpart, ascending
  tf::buffer<Index> new_faces;
  /// number of input meshes; the source axis `end` sentinel value
  Index n_sources = 0;
  Index n_output_points = 0;
  /// total output faces; doubles as `face_f`'s `end` sentinel value
  Index n_output_faces = 0;
};
} // namespace tf
