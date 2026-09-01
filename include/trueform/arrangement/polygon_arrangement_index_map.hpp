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

namespace tf {

/// Relates the output of make_polygon_arrangements back to the single input
/// mesh. All label buffers are output-indexed; point_f is the forward map.
///
/// Created points (intersection vertices with no input origin) carry the `end`
/// sentinel, `point_labels[o] == n_output_points`. Detect one positionally,
/// `o >= n_original_points`, whose created id is `o - n_original_points`.
///
/// The sentinel itself is not a safe detector: `point_labels` holds an input
/// point id, and that space is unrelated to the output's — a kept original
/// whose id happens to equal `n_output_points` would read as created. With one
/// form there is no tag axis to fall back on, so the positional test is the
/// only reliable one.
///
/// Faces have no `end` tail: a cut face is a piece of an input face, so every
/// output face keeps a real origin in face_labels.
template <typename Index> struct polygon_arrangement_index_map {
  /// output point -> input point id; created -> n_output_points
  tf::buffer<Index> point_labels;
  /// output face -> input face id (origin face for cut faces)
  tf::buffer<Index> face_labels;
  /// point_f[input id] -> output point id; unmapped -> n_output_points
  tf::buffer<Index> point_f;
  /// output points below this are kept originals, at/above are created
  Index n_original_points = 0;
  Index n_original_faces = 0;
  /// total output points; doubles as the `end` sentinel value
  Index n_output_points = 0;
};

} // namespace tf
