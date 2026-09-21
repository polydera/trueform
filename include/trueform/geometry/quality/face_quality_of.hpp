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
#include "../../core/angle.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/polygon.hpp"
#include "../../core/sqrt.hpp"
#include "../../core/static_size.hpp"
#include "../../core/triangle_quality.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tf::geometry {

/// @ingroup geometry
/// @brief One face's quality numbers, as @ref tf::face_quality states them.
template <typename T> struct face_quality_values {
  T quality;
  tf::rad<T> min_angle;
  tf::rad<T> max_angle;
  T aspect_ratio;
};

/// @ingroup geometry
/// @brief Measure one face: its triangle quality, the extremes of its corner
/// angles, and its aspect ratio.
///
/// A corner angle is the turn taken against the reach, so a sliver's vanishing
/// angle survives where a cosine would have lost it.
template <typename Policy>
auto face_quality_of(const tf::polygon<3, Policy> &face)
    -> face_quality_values<tf::coordinate_type<Policy>> {
  using T = tf::coordinate_type<Policy>;

  if constexpr (tf::static_size_v<Policy> == 3) {
    const auto e0 = face[1] - face[0];
    const auto e1 = face[2] - face[0];
    const auto e2 = face[2] - face[1];
    const T l0 = e0.length2();
    const T l1 = e1.length2();
    const T l2 = e2.length2();

    // a triangle spans one parallelogram, so all three corners turn through it
    const T two_area = tf::cross(e0, e1).length();
    const T max_e2 = std::max({l0, l1, l2});
    const T min_e2 = std::min({l0, l1, l2});

    // the shared turn is non-negative and atan2 falls as its second argument
    // rises, so the extreme corners are the extreme reaches, and the middle
    // angle is never measured
    const T r0 = tf::dot(e0, e1);
    const T r1 = -tf::dot(e0, e2);
    const T r2 = tf::dot(e1, e2);

    return {max_e2 > 0 ? tf::triangle_quality(two_area, max_e2) : T(0),
            tf::rad<T>{std::atan2(two_area, std::max({r0, r1, r2}))},
            tf::rad<T>{std::atan2(two_area, std::min({r0, r1, r2}))},
            min_e2 > 0 ? tf::sqrt(max_e2 / min_e2)
                       : std::numeric_limits<T>::infinity()};
  } else {
    auto n = face.size();
    auto incoming = face[0] - face[n - 1];
    T max_e2 = 0;
    T min_e2 = std::numeric_limits<T>::max();
    T min_angle = std::numeric_limits<T>::max();
    T max_angle = 0;

    T turn = 0;
    auto nxt = decltype(n)(1);
    for (decltype(n) i = 0; i < n; ++i) {
      const auto outgoing = face[nxt] - face[i];
      const T side2 = outgoing.length2();
      max_e2 = std::max(max_e2, side2);
      min_e2 = std::min(min_e2, side2);

      turn = tf::cross(incoming, outgoing).length();
      const T angle = std::atan2(turn, -tf::dot(incoming, outgoing));
      min_angle = std::min(min_angle, angle);
      max_angle = std::max(max_angle, angle);

      incoming = outgoing;
      nxt = nxt + 1 == n ? decltype(n)(0) : nxt + 1;
    }

    const T aspect_ratio = min_e2 > 0 ? tf::sqrt(max_e2 / min_e2)
                                      : std::numeric_limits<T>::infinity();
    if (n != 3)
      return {T(-1), tf::rad<T>{min_angle}, tf::rad<T>{max_angle},
              aspect_ratio};

    // a triangle's corners all turn through its one parallelogram, so the last
    // turn the walk measured is the doubled area
    return {max_e2 > 0 ? tf::triangle_quality(turn, max_e2) : T(0),
            tf::rad<T>{min_angle}, tf::rad<T>{max_angle}, aspect_ratio};
  }
}

} // namespace tf::geometry
