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

#include "trueform/cpp/topology/k_rings.hpp"
#include "trueform/cpp/topology/neighborhoods.hpp"

#include "trueform/core/distance.hpp"
#include "trueform/core/points.hpp"
#include "trueform/topology/make_k_rings.hpp"
#include "trueform/topology/make_neighborhoods.hpp"
#include "trueform/topology/vertex_link_like.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tf::cpp {
namespace detail {

template <typename Index>
auto require_neighborhood_connectivity(
    const offset_blocked_buffer<Index, Index> &connectivity,
    const char *operation) -> int {
  if (!connectivity.is_valid())
    throw std::invalid_argument(std::string(operation) +
                                ": connectivity must be valid");

  const auto offsets = connectivity.offsets();
  const auto data = connectivity.data();
  if (offsets.ndim() != 1 || data.ndim() != 1)
    throw std::invalid_argument(
        std::string(operation) +
        ": connectivity arrays must be one-dimensional");
  if (offsets.length() == 0) {
    if (data.length() != 0)
      throw std::invalid_argument(std::string(operation) +
                                  ": empty offsets require empty data");
    return 0;
  }
  if (offsets[0] != Index{0})
    throw std::invalid_argument(std::string(operation) +
                                ": offsets must start at zero");
  for (std::size_t i = 1; i < offsets.length(); ++i) {
    if (offsets[i] < offsets[i - 1] || offsets[i] < Index{0} ||
        static_cast<std::size_t>(offsets[i]) > data.length())
      throw std::invalid_argument(std::string(operation) +
                                  ": offsets must be ordered within data");
  }
  if (static_cast<std::size_t>(offsets[offsets.length() - 1]) != data.length())
    throw std::invalid_argument(std::string(operation) +
                                ": final offset must equal data length");
  const auto count = static_cast<int>(offsets.length() - 1);
  const auto index_count = static_cast<Index>(count);
  for (const auto peer : data) {
    if (peer < Index{0} || peer >= index_count)
      throw std::out_of_range(std::string(operation) +
                              ": peer index out of range");
  }
  return count;
}

} // namespace detail

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto k_rings(const offset_blocked_buffer<Index, Index> &connectivity,
             std::int32_t k, bool inclusive)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_neighborhood_connectivity(connectivity, "k_rings");
  if (k <= 0)
    throw std::invalid_argument("k_rings: k must be positive");

  const auto vertex_link = tf::make_vertex_link_like(connectivity.make_range());
  return offset_blocked_buffer<Index, Index>::from_buffer(
      tf::make_k_rings(vertex_link, static_cast<std::size_t>(k), inclusive));
}

template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<
              is_supported_index_v<Index> && (Dims == 2 || Dims == 3), int>>
auto neighborhoods(const offset_blocked_buffer<Index, Index> &connectivity,
                   const nd_array<Real> &points, Real radius, bool inclusive)
    -> offset_blocked_buffer<Index, Index> {
  const auto count =
      detail::require_neighborhood_connectivity(connectivity, "neighborhoods");
  if (!points.is_valid() || points.ndim() != 2 ||
      points.shape_at(1) != static_cast<int>(Dims))
    throw std::invalid_argument(
        "neighborhoods: points must have shape [N, Dims]");
  if (points.shape_at(0) != count)
    throw std::invalid_argument(
        "neighborhoods: point count must equal connectivity block count");
  if (std::isnan(radius) || radius <= Real{0})
    throw std::invalid_argument(
        "neighborhoods: radius must be positive and not NaN");

  const auto vertex_link = tf::make_vertex_link_like(connectivity.make_range());
  const auto point_range = tf::make_points<Dims>(points.make_range());
  return offset_blocked_buffer<Index, Index>::from_buffer(
      tf::make_neighborhoods(
          vertex_link,
          [&point_range](Index seed, Index neighbor) {
            return tf::distance2(point_range[seed], point_range[neighbor]);
          },
          radius, inclusive));
}

} // namespace tf::cpp
