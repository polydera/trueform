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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/reindex/selection_result.hpp"

#include <cstddef>

namespace tf::cpp::detail {

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct reindexed_fixed_carrier;

template <typename Index, typename Real, std::size_t Dims>
struct reindexed_fixed_carrier<Index, Real, Dims, 2> {
  using type = tf::segments_buffer<Index, Real, Dims>;
};

template <typename Index, typename Real, std::size_t Dims>
struct reindexed_fixed_carrier<Index, Real, Dims, 3> {
  using type = tf::polygons_buffer<Index, Real, Dims, 3>;
};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct reindexed_fixed_result;

template <typename Index, typename Real, std::size_t Dims>
struct reindexed_fixed_result<Index, Real, Dims, 2> {
  using type = reindexed_edge_mesh_result<Index, Real, Dims>;
};

template <typename Index, typename Real, std::size_t Dims>
struct reindexed_fixed_result<Index, Real, Dims, 3> {
  using type = reindexed_mesh_result<Index, Real, Dims>;
};

} // namespace tf::cpp::detail
