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
#include "../../core/coordinate_type.hpp"
#include "../../core/point.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {
namespace volume_detail {

/// @brief The grid a volume's samples stand on: the samples range itself, plus
/// the metadata that names them.
///
/// The storage policy behind @ref tf::volume, as `tf::core::pt` is the one
/// behind @ref tf::point_like. It IS the range, so a volume iterates its own
/// samples, and it publishes `coordinate_dims`, so every reader deduces how
/// many axes the grid has through @ref tf::coordinate_dims_v rather than being
/// told. `Coord` is the domain scalar — spacing, origin, and every position
/// answer in it; the range's own scalar remains `sample_type`.
template <typename Range, typename Coord, std::size_t Dims>
struct grid : Range {
  using sample_type = tf::coordinate_type<Range>;
  using coordinate_type = Coord;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  grid() = default;

  grid(const Range &samples, const std::array<int, Dims> &dims,
       const tf::point<Coord, Dims> &spacing,
       const tf::point<Coord, Dims> &origin)
      : Range{samples}, _dims(dims), _spacing(spacing), _origin(origin) {}

  grid(Range &&samples, const std::array<int, Dims> &dims,
       const tf::point<Coord, Dims> &spacing,
       const tf::point<Coord, Dims> &origin)
      : Range{std::move(samples)}, _dims(dims), _spacing(spacing),
        _origin(origin) {}

  /// @brief Number of samples along each axis.
  auto dims() const -> const std::array<int, Dims> & { return _dims; }

  /// @brief Physical size of one grid step along each axis.
  auto spacing() const -> const tf::point<Coord, Dims> & { return _spacing; }

  /// @brief Local-space position of sample (0, ..., 0).
  auto origin() const -> const tf::point<Coord, Dims> & { return _origin; }

private:
  std::array<int, Dims> _dims{};
  tf::point<Coord, Dims> _spacing;
  tf::point<Coord, Dims> _origin;
};

} // namespace volume_detail
} // namespace tf
