/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/offset_block_buffer.hpp"

namespace tf::cut {
template <typename Index> struct polygon_arrangement_ids {
  tf::offset_block_buffer<Index, Index> polygons;
  tf::offset_block_buffer<Index, Index> cut_faces;
};
} // namespace tf::cut
