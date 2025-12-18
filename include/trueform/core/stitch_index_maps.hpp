/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./index_map.hpp"

namespace tf {
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
};
} // namespace tf
