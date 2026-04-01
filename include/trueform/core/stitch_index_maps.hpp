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
#include "./index_map.hpp"
#include "./direction.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Index maps for stitching two polygon meshes.
///
/// Contains mappings for points and polygons from both input meshes,
/// plus any newly created points during stitching. Tracks the
/// iteration direction used for each mesh's boundary.
///
/// @tparam Index The index type.
template <typename Index> struct stitch_index_maps {
  tf::index_map_buffer<Index> points0;
  Index points0_offset;
  tf::index_map_buffer<Index> points1;
  Index points1_offset;
  tf::index_map_buffer<Index> created_points;
  Index created_points_offset;
  tf::index_map_buffer<Index> polygons0;
  Index polygons0_offset;
  tf::index_map_buffer<Index> polygons1;
  Index polygons1_offset;
  tf::direction direction0;
  tf::direction direction1;
};
} // namespace tf
