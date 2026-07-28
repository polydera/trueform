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

#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "./lift_region_point_to_plane.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace tf::cut {

/// The interior points refinement chose for a region: kept by a face the
/// triangulation retains, on no constraint, and not one of the points the
/// region entered with. `flags` is reusable scratch; `points` is appended
/// to, never cleared, so a caller accumulating across regions records its
/// own watermark.
template <typename Index, typename Int>
auto collect_region_steiner_points(
    const tf::cut::detail::region_triangulation_workspace<Index, Int>
        &workspace,
    const tf::cdt_refiner<Index, Int, Int> &refiner, tf::buffer<char> &flags,
    tf::buffer<tf::point<Int, 3>> &points) -> void {
  constexpr char used = 1;
  constexpr char on_constraint = 2;
  constexpr char is_input = 4;

  const auto refined_points = refiner.points();
  flags.allocate(refined_points.size());
  std::fill(flags.begin(), flags.end(), char(0));

  const auto labels = refiner.region_labels();
  for (Index face = 0; face < refiner.n_faces(); ++face) {
    if ((labels[std::size_t(face)] & 1) == 0)
      continue;
    const auto triangle = refiner.face(face);
    for (int edge = 0; edge < 3; ++edge) {
      flags[std::size_t(triangle[std::size_t(edge)])] |= used;
      if (refiner.constrained(face, edge)) {
        flags[std::size_t(triangle[std::size_t(edge)])] |= on_constraint;
        flags[std::size_t(triangle[std::size_t((edge + 1) % 3)])] |=
            on_constraint;
      }
    }
  }

  const auto &input_to_output = refiner.index_map().f();
  const Index n_input = refiner.n_input_points();
  for (Index input = 0; input < n_input; ++input) {
    const Index output = input_to_output[std::size_t(input)];
    if (std::make_signed_t<Index>(output) >= 0 &&
        output < Index(refined_points.size()))
      flags[std::size_t(output)] |= is_input;
  }

  for (std::size_t point = 0; point < refined_points.size(); ++point)
    if (flags[point] == used)
      points.push_back(tf::cut::lift_region_point_to_plane(
          workspace, refined_points[point]));
}

} // namespace tf::cut
