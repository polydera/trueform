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

#include "trueform/core/points.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/unit_vectors.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::cpp {

/// @brief One reading of a point cloud, and the stamp that names it.
///
/// The mesh geometry's contract, for the point carrier — see
/// `trueform/cpp/core/mesh_geometry.hpp`. A cloud has one array and one stamp,
/// and the normals ride with it: they are a per-point attribute, so they name
/// the points this reading holds and not whatever the caller holds by now.
template <typename Real, std::size_t Dims> struct point_cloud_geometry {
  using real_range = tf::range<const Real *, tf::dynamic_size>;
  using points_type = decltype(tf::make_points<Dims>(
      std::declval<tf::range<const Real *, tf::dynamic_size>>()));
  using normals_type = decltype(tf::make_unit_vectors<Dims>(
      std::declval<tf::range<const Real *, tf::dynamic_size>>()));

  real_range coordinates;
  real_range normal_coordinates;
  std::uint64_t points_stamp = 0;

  auto points() const -> points_type {
    return tf::make_points<Dims>(coordinates);
  }

  auto normals() const -> normals_type {
    return tf::make_unit_vectors<Dims>(normal_coordinates);
  }

  auto has_normals() const -> bool { return normal_coordinates.size() != 0; }

  auto number_of_points() const -> std::size_t {
    return coordinates.size() / Dims;
  }
};

} // namespace tf::cpp
