/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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
