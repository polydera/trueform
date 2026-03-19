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

#include "../core/buffer.hpp"
#include "../core/small_vector.hpp"
#include "../intersect/graph/vertex.hpp"

namespace tf::cut {

/// Internal data produced by build_arrangement_maps.
/// Holds vertex ID maps, face masks, and offset tables needed
/// to assemble the final mesh arrangement.
template <typename Index> struct arrangement_map_data {
  using vertex_t = tf::intersect::graph::vertex<Index>;

  // Per-mesh original vertex IDs (local mesh indices of used vertices)
  tf::small_vector<tf::buffer<Index>, 10> original_ids;
  // Per-mesh uncut face indices
  tf::small_vector<tf::buffer<Index>, 10> original_face_ids;
  // Flat map: old_flat_id → local_new_id (per mesh, before prefix sum)
  tf::buffer<Index> original_map;
  // Prefix sum of per-mesh used original vertex counts (output point offsets)
  tf::buffer<Index> original_offsets;
  // Prefix sum of per-mesh uncut face counts (output face offsets)
  tf::buffer<Index> original_face_offsets;
  // Cumulative mesh point counts (for flat indexing into original_map)
  tf::buffer<Index> point_offsets;
  // Cumulative mesh face counts (for make_face_index)
  tf::buffer<Index> poly_offsets;
  // Number of input meshes
  Index n_meshes;
  // Total used original vertices (across all meshes)
  Index total_original_points;
  // Total uncut faces (across all meshes)
  Index total_original_faces;

  auto map_vertex(Index tag, const vertex_t &v) const -> Index {
    if (v.source == tf::intersect::graph::vertex_source::original)
      return original_map[point_offsets[tag] + v.id] + original_offsets[tag];
    return v.id + total_original_points;
  }

  auto map_original_vertex(Index tag, Index v) -> Index {
    return original_map[point_offsets[tag] + v.id] + original_offsets[tag];
  }
};

} // namespace tf::cut
