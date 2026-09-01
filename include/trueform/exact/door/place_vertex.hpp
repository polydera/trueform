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
#include "../../core/vector.hpp"
#include "../meta.hpp"
#include "./admits_placement.hpp"
#include "./place_at_meet.hpp"
#include "./place_on_line.hpp"
#include "./place_on_plane.hpp"
#include "./plane_frame.hpp"
#include "./plane_step.hpp"
#include "./quantize_direction.hpp"
#include "./round_div.hpp"
#include "./round_to_wide.hpp"
#include "./wide_cross_magnitude.hpp"
#include "./wide_det3.hpp"
#include "./wide_dot.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// Where a vertex stands after the door, and by which rank it got
/// there. Rank 0 is the vertex the door left alone.
template <typename Int> struct vertex_placement {
  tf::point<Int, 3> point{};
  int rank = 0;
};

/// Where one vertex stands after the door. The candidates are the plane
/// names its incident faces state, deduplicated and in name order — the
/// caller states them, so the answer is a pure function of that list,
/// of the vertex and of the tolerance. The ranks are tried in order:
/// the meet of the most independent triple, then the most independent
/// pair of that triple, then the vertex's own tangent plane. Each is
/// admitted only by @ref tf::exact::door::admits_placement; the last
/// one failing leaves the vertex where it was.
///
/// Rank 3 is a corner whose three directions survive the grid, and two
/// forms meeting at such a corner land on one integer because rank 3
/// reads names and nothing else. Rank 1 is the smooth case: the vertex
/// lands exactly on its own quantized tangent plane, which is what
/// keeps a coplanar group coplanar.
template <typename Int, typename Candidates>
auto place_vertex(const tf::point<Int, 3> &original,
                  const Candidates &candidates,
                  const tf::vector<double, 3> &normal, Int tolerance)
    -> vertex_placement<Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  vertex_placement<Int> placement{original, 0};
  if (tolerance <= Int(0))
    return placement;

  const auto accept = [&placement](const std::array<T1, 3> &at, int rank) {
    placement.point = tf::point<Int, 3>{static_cast<Int>(at[0]),
                                        static_cast<Int>(at[1]),
                                        static_cast<Int>(at[2])};
    placement.rank = rank;
  };

  const std::size_t n = candidates.size();
  std::size_t i0 = 0, i1 = 0, i2 = 0;
  bool triple = false;
  if (n >= 3) {
    T2 widest(0);
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = a + 1; b < n; ++b)
        for (std::size_t c = b + 1; c < n; ++c) {
          T2 measure = wide_det3<Int>(candidates[a].normal, candidates[b].normal,
                                      candidates[c].normal);
          if (measure < T2(0))
            measure = -measure;
          if (measure > widest) {
            widest = measure;
            i0 = a;
            i1 = b;
            i2 = c;
            triple = true;
          }
        }
  }

  std::array<T1, 3> at{};
  if (triple &&
      place_at_meet(candidates[i0], candidates[i1], candidates[i2], at) &&
      admits_placement(original, at, tolerance)) {
    accept(at, 3);
    return placement;
  }

  if (n >= 2) {
    std::size_t j0 = 0, j1 = 0;
    T2 widest(0);
    bool pair = false;
    const auto consider = [&](std::size_t a, std::size_t b) {
      const T2 measure =
          wide_cross_magnitude<Int>(candidates[a].normal, candidates[b].normal);
      if (measure > widest) {
        widest = measure;
        j0 = a;
        j1 = b;
        pair = true;
      }
    };
    if (triple) {
      consider(i0, i1);
      consider(i0, i2);
      consider(i1, i2);
    } else {
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b)
          consider(a, b);
    }
    if (pair && place_on_line(candidates[j0], candidates[j1], original, at) &&
        admits_placement(original, at, tolerance)) {
      accept(at, 2);
      return placement;
    }
  }

  std::array<T1, 3> tangent{};
  if (quantize_direction<Int>(normal[0], normal[1], normal[2],
                              T1(tolerance), tangent)) {
    const T1 step = plane_step<Int>(tangent, T1(tolerance));
    const T2 height = wide_dot<Int>(tangent, original);
    const T2 offset = round_div(height, T2(step)) * T2(step);
    const T2 residual = offset - height;
    const T2 bound = T2(wide_placement_bound<Int>());
    std::array<T1, 3> shift{};
    if (residual <= bound && residual >= -bound &&
        place_on_plane<Int>(make_plane_frame<Int>(tangent),
                            static_cast<T1>(residual), shift)) {
      at = {T1(original[0]) + shift[0], T1(original[1]) + shift[1],
            T1(original[2]) + shift[2]};
      if (admits_placement(original, at, tolerance)) {
        accept(at, 1);
        return placement;
      }
    }
  }
  return placement;
}

} // namespace tf::exact::door
