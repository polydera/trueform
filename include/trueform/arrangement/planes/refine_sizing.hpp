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

#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace tf::arrangement {

/// How fast the sizing field may grow between adjacent points of a cycle.
inline constexpr double k_sizing_lipschitz = 1.0;
/// Slack on the adjacent-edge term, so a cycle is not sized to its own
/// shortest edge exactly.
inline constexpr double k_edge_length_relaxation = 1.5;
/// An interval splits when it is longer than this multiple of the sizing
/// field at its midpoint.
inline constexpr double k_split_length_ratio = 1.5;
/// The local-feature-size term compares every point against every segment,
/// so it is only affordable below this cycle length.
inline constexpr std::size_t k_local_feature_size_limit = 2048;

/// Distance from `query` to the segment `first`-`second`, in 2D.
inline auto distance_to_segment(const std::array<double, 2> &query,
                                const std::array<double, 2> &first,
                                const std::array<double, 2> &second)
    -> double {
  const double delta_x = second[0] - first[0];
  const double delta_y = second[1] - first[1];
  const double length2 = delta_x * delta_x + delta_y * delta_y;
  double parameter = 0;
  if (length2 > 0)
    parameter = std::clamp(((query[0] - first[0]) * delta_x +
                            (query[1] - first[1]) * delta_y) /
                               length2,
                           0.0, 1.0);
  const double offset_x = first[0] + parameter * delta_x - query[0];
  const double offset_y = first[1] + parameter * delta_y - query[1];
  return std::sqrt(offset_x * offset_x + offset_y * offset_y);
}

/// Sizing field on a closed cycle of 2D points: the shorter adjacent edge at
/// each point, optionally narrowed to the distance to the nearest
/// non-adjacent segment, then smoothed cyclically in both directions so the
/// field never grows faster than @ref k_sizing_lipschitz along the cycle.
template <typename Points>
inline auto cycle_sizing(const Points &points, bool with_local_feature_size,
                         tf::buffer<double> &sizing,
                         tf::buffer<double> &edge_lengths) -> void {
  const std::size_t size = points.size();
  edge_lengths.allocate(size);
  for (std::size_t i = 0; i < size; ++i) {
    const auto &first = points[i];
    const auto &second = points[(i + 1) % size];
    edge_lengths[i] = std::hypot(second[0] - first[0], second[1] - first[1]);
  }
  sizing.allocate(size);
  for (std::size_t i = 0; i < size; ++i)
    sizing[i] = k_edge_length_relaxation *
                std::min(edge_lengths[i], edge_lengths[(i + size - 1) % size]);
  if (with_local_feature_size && size <= k_local_feature_size_limit) {
    for (std::size_t i = 0; i < size; ++i) {
      const std::size_t previous = (i + size - 1) % size;
      for (std::size_t j = 0; j < size; ++j) {
        if (j == i || j == previous)
          continue;
        const double distance =
            distance_to_segment(points[i], points[j], points[(j + 1) % size]);
        if (distance > 0)
          sizing[i] = std::min(sizing[i], distance);
      }
    }
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < size; ++i) {
      const std::size_t next = (i + 1) % size;
      sizing[next] = std::min(sizing[next],
                              sizing[i] + k_sizing_lipschitz * edge_lengths[i]);
    }
    for (std::size_t i = size; i-- > 0;) {
      const std::size_t previous = (i + size - 1) % size;
      sizing[previous] = std::min(
          sizing[previous],
          sizing[i] + k_sizing_lipschitz * edge_lengths[previous]);
    }
  }
}

/// Split the parameter interval `[begin, end)` recursively wherever it is
/// long relative to the sizing field, reporting each split parameter through
/// `emit`. The sizing at a midpoint is the lesser of the interpolated value
/// and what `local_feature_size_at` measures there.
///
/// Every emitted parameter stays a multiple of four, so the same interval
/// splits to the same parameters whichever side of a shared edge asks —
/// which is what lets two carriers of one edge agree without talking.
template <typename Emit, typename LocalFeatureSizeAt>
inline auto dyadic_split(std::uint32_t begin, std::uint32_t end,
                         const std::array<double, 2> &begin_point,
                         const std::array<double, 2> &end_point,
                         double begin_sizing, double end_sizing, Emit &&emit,
                         LocalFeatureSizeAt &&local_feature_size_at) -> void {
  if (end - begin <= 4)
    return;
  const std::uint32_t middle = (begin + end) / 2;
  if (middle & 3)
    return;
  const std::array<double, 2> middle_point{0.5 * (begin_point[0] + end_point[0]),
                                           0.5 *
                                               (begin_point[1] + end_point[1])};
  const double length = std::hypot(end_point[0] - begin_point[0],
                                   end_point[1] - begin_point[1]);
  const double middle_sizing = std::min(0.5 * (begin_sizing + end_sizing),
                                        local_feature_size_at(middle_point));
  if (length <= k_split_length_ratio * middle_sizing)
    return;
  emit(middle);
  dyadic_split(begin, middle, begin_point, middle_point, begin_sizing,
               middle_sizing, emit, local_feature_size_at);
  dyadic_split(middle, end, middle_point, end_point, middle_sizing, end_sizing,
               emit, local_feature_size_at);
}

} // namespace tf::arrangement
