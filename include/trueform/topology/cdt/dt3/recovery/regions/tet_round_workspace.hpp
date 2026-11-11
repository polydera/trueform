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
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/memory.hpp"
#include "../../tet_round_state.hpp"
#include "../../tet_row_pool.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The workspace one round of the regions walks on, built once by the phase
/// that owns it and passed in: the bucket's bounds, order and cursor, the
/// flag per due item, the pools, each region's capped walks and hull bars,
/// and one state per region. Its lifetime is
/// the phase; a round clears what it fills.
template <typename Index> struct tet_round_workspace {
  tf::buffer<Index> offsets;
  tf::buffer<Index> order;
  tf::buffer<Index> cursor;
  tf::buffer<char> done;
  tf::buffer<tet_row_pool> pools;
  tf::buffer<std::array<std::size_t, 2>> bars;
  tf::core::std_vector<tet_round_state<Index>> states;
};

} // namespace tf::topology::cdt::dt3
