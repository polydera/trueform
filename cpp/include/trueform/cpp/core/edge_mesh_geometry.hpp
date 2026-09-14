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

#include "trueform/core/edges.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/views/blocked_range.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::cpp {

/// @brief One reading of an edge mesh, and the two stamps that name it.
///
/// The edge carrier's spelling of `mesh_geometry` — see
/// `trueform/cpp/core/mesh_geometry.hpp`.
template <typename Index, typename Real, std::size_t Dims>
struct edge_mesh_geometry {
  using index_range = tf::range<const Index *, tf::dynamic_size>;
  using real_range = tf::range<const Real *, tf::dynamic_size>;
  using edges_type = decltype(tf::make_edges(
      tf::make_blocked_range<2>(std::declval<index_range>())));
  using points_type = decltype(tf::make_points<Dims>(
      std::declval<tf::range<const Real *, tf::dynamic_size>>()));

  index_range indices;
  real_range coordinates;
  std::uint64_t edges_stamp = 0;
  std::uint64_t points_stamp = 0;

  auto edges() const -> edges_type {
    return tf::make_edges(tf::make_blocked_range<2>(indices));
  }

  auto points() const -> points_type {
    return tf::make_points<Dims>(coordinates);
  }

  auto number_of_edges() const -> std::size_t { return indices.size() / 2; }

  auto number_of_points() const -> std::size_t {
    return coordinates.size() / Dims;
  }
};

} // namespace tf::cpp
