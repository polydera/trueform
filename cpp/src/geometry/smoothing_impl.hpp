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

#include "trueform/cpp/geometry/laplacian_smoothed.hpp"
#include "trueform/cpp/geometry/taubin_smoothed.hpp"

#include "../core/materialized_mesh.hpp"

#include "trueform/core/points.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/geometry/laplacian_smoothed.hpp"
#include "trueform/geometry/taubin_smoothed.hpp"
#include "trueform/topology/policy/vertex_link.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace tf::cpp {
namespace detail {

/// Smoothing moves points and says nothing about connectivity, so the output is
/// this mesh's faces with the points the smoother computed.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Function>
auto smooth(const mesh<Index, Real, Dims, Ngon> &value, int iterations,
            const char *operation, Function &&function)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  if (iterations < 0)
    throw std::invalid_argument(std::string(operation) +
                                ": iterations must be non-negative");

  auto tagged_points = value.points() | tf::tag(value.vertex_link());
  auto smoothed = function(tagged_points);

  tf::polygons_buffer<Index, Real, Dims, Ngon> output;
  detail::materialize_faces(value, output);
  output.points_buffer().data_buffer() = std::move(smoothed.data_buffer());
  return output;
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto laplacian_smoothed(const mesh<Index, Real, Dims, Ngon> &value,
                        int iterations, Real lambda)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return detail::smooth(value, iterations, "laplacian_smoothed",
                        [iterations, lambda](const auto &points) {
                          return tf::laplacian_smoothed(
                              points, static_cast<std::size_t>(iterations),
                              lambda);
                        });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto taubin_smoothed(const mesh<Index, Real, Dims, Ngon> &value, int iterations,
                     Real lambda, Real kpb)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return detail::smooth(value, iterations, "taubin_smoothed",
                        [iterations, lambda, kpb](const auto &points) {
                          return tf::taubin_smoothed(
                              points, static_cast<std::size_t>(iterations),
                              lambda, kpb);
                        });
}

} // namespace tf::cpp
