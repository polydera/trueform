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

#include "../../core/point.hpp"
#include "../meta.hpp"
#include "../plane_support.hpp"
#include "./plane_step.hpp"
#include "./quantize_direction.hpp"
#include "./quantized_plane.hpp"
#include "./round_div.hpp"
#include "./round_to_wide.hpp"

#include <array>

namespace tf::exact::door {

/// The plane a face names on the door's grid. The direction is the
/// face's own exact normal quantized; the offset is quantized at the
/// face's centroid, because a face has no vertex and only the
/// symmetric choice keeps the name the face's — an offset taken at one
/// corner would make the same plane unshareable between the faces that
/// lie in it, and pooling would be lost before the placement starts.
///
/// False is the verdict for a face that states no plane: fewer than
/// three points off one line, a direction that rounds to zero, or an
/// offset outside the rung the meet is solved on.
template <typename Int, typename Corners>
auto quantize_face_plane(const Corners &corners, Int tolerance,
                         quantized_plane<Int> &plane) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  tf::exact::plane_support<Int> support;
  for (const auto &corner : corners) {
    if (support.size == 3)
      break;
    support.offer(tf::point<Int, 3>{corner[0], corner[1], corner[2]});
  }
  if (support.size < 3)
    return false;

  std::array<T1, 3> normal{};
  if (!quantize_direction<Int>(static_cast<double>(support.normal[0]),
                               static_cast<double>(support.normal[1]),
                               static_cast<double>(support.normal[2]),
                               T1(tolerance), normal))
    return false;

  const T1 step = plane_step<Int>(normal, T1(tolerance));
  T2 sum(0);
  T2 count(0);
  for (const auto &corner : corners) {
    sum = sum + T2(normal[0]) * T2(corner[0]) + T2(normal[1]) * T2(corner[1]) +
          T2(normal[2]) * T2(corner[2]);
    count = count + T2(1);
  }
  const T2 offset = round_div(sum, count * T2(step)) * T2(step);
  const T2 bound = T2(wide_placement_bound<Int>());
  if (offset > bound || offset < -bound)
    return false;

  plane.normal = normal;
  plane.offset = static_cast<T1>(offset);
  return true;
}

} // namespace tf::exact::door
