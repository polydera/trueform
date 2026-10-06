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
#include "../../../core/buffer.hpp"
#include "../../../core/small_vector.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index> struct tet_edge_query_scratch {
  tf::buffer<std::uint32_t> visited;
  tf::small_vector<Index, 64> stack;
  std::uint32_t epoch = 0;
};
template <typename Index> struct tet_edge_query_workspace {
  tf::buffer<Index> incident;
  tf::buffer<char> present;
  tf::buffer<Index> missing;
};

} // namespace tf::topology::cdt::dt3
