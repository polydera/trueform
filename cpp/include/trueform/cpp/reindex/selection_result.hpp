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
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Owning mesh-selection result and source maps.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct reindexed_mesh_result {
  tf::polygons_buffer<Index, Real, Dims, Ngon> mesh;
  index_map<Index> face_map;
  index_map<Index> point_map;
};

/// @brief Typed owning edge-mesh-selection result and source maps.
template <typename Index, typename Real, std::size_t Dims>
struct reindexed_edge_mesh_result {
  tf::segments_buffer<Index, Real, Dims> mesh;
  index_map<Index> edge_map;
  index_map<Index> point_map;
};

/// @brief Typed owning raw-point selection result and source map.
template <typename Index, typename Real> struct reindexed_points_result {
  nd_array<Real> points;
  index_map<Index> point_map;
};

} // namespace tf::cpp
