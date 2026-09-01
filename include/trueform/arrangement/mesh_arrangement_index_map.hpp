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

namespace tf {

/// Relates the output of make_mesh_arrangements back to the N input meshes.
/// Adds the mesh-tag axis to the single-mesh case: output labels carry both
/// which input mesh (`*_tag_labels`) and which input element (`*_labels`), and
/// the forward map is per mesh (`point_f[tag][input id]`).
///
/// Created points (intersection vertices with no input origin) carry the `end`
/// sentinel on both axes: the tag ends at `n_tags`, the point id at
/// `n_output_points`. Detect one by the **tag**, `point_tag_labels[o] ==
/// n_tags`, or positionally, `o >= n_original_points` — whose created id is
/// `o - n_original_points`. Both are always reliable: valid tags are
/// `[0, n_tags)`, and originals precede created points.
///
/// The point-id sentinel is not a safe detector. `point_labels` holds an input
/// point id local to its own form, and that space is unrelated to the output's
/// — a kept original whose local id happens to equal `n_output_points` would
/// read as created.
///
/// Faces have no `end` tail: a cut face is a piece of an input face, so every
/// output face keeps a real (tag, origin face).
///
/// `uncut_faces[tag]` is the `[begin, end)` output range of tag `tag`'s
/// uncut faces — each one still the entire input face `face_labels[o]`
/// rather than a piece of it. Every face outside those ranges is a cut
/// piece. Consumers that carry per-face data across the arrangement (edge
/// links, memberships) need exactly this split: an uncut face can be
/// copied, a cut one has to be recomputed. It is a per-tag range and not a
/// single prefix because the producers order faces differently — the
/// arrangement groups every tag's uncut faces first and puts all cut
/// triangles after them, while the boolean emits each tag's uncut faces
/// then its own cut ones before moving to the next tag. Both keep a tag's
/// uncut faces contiguous, which is all this records.
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
  /// uncut_faces[tag] = {begin, end} of tag's uncut output faces
  tf::blocked_buffer<Index, 2> uncut_faces;
  /// output points below this are kept originals, at/above are created
  Index n_original_points = 0;
  /// number of input meshes; the tag axis `end` sentinel value
  Index n_tags = 0;
  /// total output points; the point-id and forward-map `end` sentinel value
  Index n_output_points = 0;
};

} // namespace tf
