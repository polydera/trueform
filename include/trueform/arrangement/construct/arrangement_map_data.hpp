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

#include "../../core/buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../intersect/graph/vertex.hpp"

namespace tf::arrangement {

/// Base vertex remapping data for mesh arrangement construction.
///
/// Maps original and created vertex IDs to compact contiguous IDs — the
/// whole output point space, originals first. It is the filtered path's
/// whole map data; the take-all path extends it with face maps.
template <typename Index> struct arrangement_point_map_data {
  using vertex_t = tf::intersect::graph::vertex<Index>;

  // Per-mesh original vertex IDs (local mesh indices of used vertices)
  tf::small_vector<tf::buffer<Index>, 10> original_ids;
  // Flat map: old_flat_id → local_new_id (per mesh, before prefix sum)
  tf::buffer<Index> original_map;
  // Prefix sum of per-mesh used original vertex counts (output point offsets)
  tf::buffer<Index> original_offsets;
  // Cumulative mesh point counts (for flat indexing into original_map)
  tf::buffer<Index> point_offsets;
  // Number of input meshes
  Index n_meshes;
  // Total used original vertices (across all meshes)
  Index total_original_points;

  // Created discovery (mark + prefix in id order): only referenced
  // created ids get output slots — a weld-retired id is never copied.
  tf::buffer<Index> created_map;
  // Which created vertex IDs are used (in encounter order)
  tf::buffer<Index> created_ids;
  // Total used created vertices
  Index total_created_points;

  /// An original stream vertex names its FLAT id — the only
  /// tag-unambiguous form inside a cross-tag pooled carrier.
  auto map_original_vertex(Index tag, Index flat) const -> Index {
    return original_map[flat] + original_offsets[tag];
  }

  auto map_vertex(Index tag, const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return map_original_vertex(tag, v.id);
    return created_map[v.id] + total_original_points;
  }
};

/// Full remapping data including face ID maps.
///
/// Extends arrangement_point_map_data with per-mesh uncut face indices
/// and offset tables. Used by the full-arrangement construction path.
template <typename Index>
struct arrangement_map_data : arrangement_point_map_data<Index> {
  // Per-mesh uncut face indices
  tf::small_vector<tf::buffer<Index>, 10> original_face_ids;
  // Prefix sum of per-mesh uncut face counts (output face offsets)
  tf::buffer<Index> original_face_offsets;
  // Cumulative mesh face counts (for make_face_index)
  tf::buffer<Index> poly_offsets;
  // Total uncut faces (across all meshes)
  Index total_original_faces;
};

} // namespace tf::arrangement
