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

namespace tf::cut {

/// Base vertex remapping data for mesh arrangement construction.
///
/// Maps original vertex IDs to compact contiguous IDs. Shared by both
/// arrangement_map_data (take all) and partition_map_data (filtered).
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

  auto map_original_vertex(Index tag, Index v) const -> Index {
    return original_map[point_offsets[tag] + v] + original_offsets[tag];
  }
};

/// Full remapping data including face ID maps.
///
/// Extends arrangement_point_map_data with per-mesh uncut face indices
/// and offset tables. Used by the "take all" construction path where
/// face IDs are discovered during map building. All created vertices
/// are used — mapped by direct offset after originals.
template <typename Index>
struct arrangement_map_data : arrangement_point_map_data<Index> {
  using vertex_t = typename arrangement_point_map_data<Index>::vertex_t;

  // Per-mesh uncut face indices
  tf::small_vector<tf::buffer<Index>, 10> original_face_ids;
  // Prefix sum of per-mesh uncut face counts (output face offsets)
  tf::buffer<Index> original_face_offsets;
  // Cumulative mesh face counts (for make_face_index)
  tf::buffer<Index> poly_offsets;
  // Total uncut faces (across all meshes)
  Index total_original_faces;

  auto map_vertex(Index tag, const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return this->map_original_vertex(tag, v.id);
    return v.id + this->total_original_points;
  }
};

/// Filtered remapping data with compact created vertex mapping.
///
/// Extends arrangement_point_map_data with created vertex compaction.
/// Used by the "filtered" construction path where only selected
/// components are included, so not all created vertices are used.
/// Face IDs come from partition_ids — not stored here.
template <typename Index>
struct partition_map_data : arrangement_point_map_data<Index> {
  using vertex_t = typename arrangement_point_map_data<Index>::vertex_t;

  // Compact map: old created id → new created id
  tf::buffer<Index> created_map;
  // Which created vertex IDs are used (in encounter order)
  tf::buffer<Index> created_ids;
  // Total used created vertices
  Index total_created_points;

  auto map_vertex(Index tag, const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return this->map_original_vertex(tag, v.id);
    return created_map[v.id] + this->total_original_points;
  }
};

/// Single-mesh take-all remapping data.
///
/// For embedding intersection curves into one mesh: takes all faces,
/// remaps original + created vertices to contiguous IDs.
template <typename Index> struct embed_map_data {
  using vertex_t = tf::intersect::graph::vertex<Index>;

  tf::buffer<Index> original_ids;
  tf::buffer<Index> original_map;
  tf::buffer<Index> uncut_face_ids;
  Index n_original_points;

  auto map_vertex(const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return original_map[v.id];
    return v.id + n_original_points;
  }
};

/// Per-mesh remapping for independent half construction.
///
/// Inherits arrangement_point_map_data but map_vertex indexes
/// original_map directly (no cross-mesh offset accumulation).
/// original_offsets[t+1] holds the per-tag count, not prefix sum.
/// Adds per-mesh created vertex compaction. Used by make_boolean_pair.
template <typename Index>
struct splice_map_data : arrangement_point_map_data<Index> {
  using vertex_t = typename arrangement_point_map_data<Index>::vertex_t;

  tf::small_vector<tf::buffer<Index>, 4> created_maps;
  tf::small_vector<tf::buffer<Index>, 4> created_ids;
  tf::small_vector<Index, 4> created_counts;

  auto map_vertex(Index tag, const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return this->original_map[this->point_offsets[tag] + v.id];
    return created_maps[tag][v.id] +
           static_cast<Index>(this->original_ids[tag].size());
  }
};

} // namespace tf::cut
