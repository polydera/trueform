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

#include "../core/point.hpp"
#include "./meta.hpp"
#include "./orient3d.hpp"
#include <array>
#include <cstddef>

namespace tf::exact {

/// The carrier plane, exact: the axes the triangulation works in, and the
/// supporting plane a 2D point lifts back through.
template <typename Int> struct plane_frame {
  int ax0 = 0;
  int ax1 = 1;
  std::array<typename tf::exact::meta<Int>::T2, 3> plane_n{};
  typename tf::exact::meta<Int>::T2 plane_d{};
  Int plane_fallback{};
};

/// A point of the frame's projection, back on the plane it came from.
/// The supporting plane is the only authority for the third coordinate;
/// a plane degenerate along the dropped axis answers with the fallback.
template <typename Int>
auto lift_plane_point(const tf::exact::plane_frame<Int> &frame,
                      const tf::point<Int, 2> &point) -> tf::point<Int, 3> {
  using T2 = typename tf::exact::meta<Int>::T2;
  const int axis_2 = 3 - frame.ax0 - frame.ax1;
  tf::point<Int, 3> lifted;
  lifted[frame.ax0] = point[0];
  lifted[frame.ax1] = point[1];
  const auto numerator = frame.plane_d -
                         frame.plane_n[std::size_t(frame.ax0)] * T2(point[0]) -
                         frame.plane_n[std::size_t(frame.ax1)] * T2(point[1]);
  const auto denominator = frame.plane_n[std::size_t(axis_2)];
  lifted[axis_2] =
      denominator == T2(0)
          ? frame.plane_fallback
          : static_cast<Int>(tf::exact::div_round(numerator, denominator));
  return lifted;
}

} // namespace tf::exact
