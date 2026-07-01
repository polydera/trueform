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

#include "../core/buffer.hpp"
#include "../core/offset_block_buffer.hpp"

namespace tf {

/// Relates the output of make_mesh_arrangements back to the N input meshes.
/// Adds the mesh-tag axis to the single-mesh case: output labels carry both
/// which input mesh (`*_tag_labels`) and which input element (`*_labels`), and
/// the forward map is per mesh (`point_f[tag][input id]`).
///
/// Created points (intersection vertices with no input origin) follow the
/// index-map `end` convention, with each axis ending past its own range: the
/// tag ends at `n_tags` (`point_tag_labels[o] == n_tags`) and the point id
/// ends at `n_output_points` (`point_labels[o] == n_output_points`). Either
/// marks a created point; equivalently `o >= n_original_points`, whose created
/// id is positional (`o - n_original_points`).
///
/// Faces have no `end` tail: a cut face is a piece of an input face, so every
/// output face keeps a real (tag, origin face).
template <typename Index> struct mesh_arrangement_index_map {
  /// output point -> input mesh tag; created -> n_tags
  tf::buffer<Index> point_tag_labels;
  /// output point -> input point id (local to its mesh); created ->
  /// n_output_points
  tf::buffer<Index> point_labels;
  /// output face -> input mesh tag
  tf::buffer<Index> face_tag_labels;
  /// output face -> input face id (origin face for cut faces)
  tf::buffer<Index> face_labels;
  /// point_f[tag][input id] -> output point id (offsets baked global). Moved
  /// straight from the construct's original_map / point_offsets.
  tf::offset_block_buffer<Index, Index> point_f;
  /// output points below this are kept originals, at/above are created
  Index n_original_points = 0;
  Index n_original_faces = 0;
  /// number of input meshes; the tag axis `end` sentinel value
  Index n_tags = 0;
  /// total output points; the point-id and forward-map `end` sentinel value
  Index n_output_points = 0;
};

} // namespace tf
