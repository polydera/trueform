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
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt {

template <typename Owner, typename F>
auto for_each_constrained_delaunay_refinement_steiner(Owner &owner, F &&f)
    -> void {
  using Index = typename Owner::index_type;

  const char used = 1;
  const char on_constraint = 2;
  const char is_input = 4;

  const auto n_points = owner._ip.size();
  owner._steiner_point_flags.allocate(n_points);
  const auto n_faces = static_cast<Index>(owner._t.size());
  const auto *const triangle_data = owner._t.begin();
  const auto *const label_data = owner._label.begin();
  const auto n_input_points = owner._n_input_points;
  const auto *const input_map_data = owner._cdt.index_map().f().begin();
  const auto *const point_data = owner._ip.begin();
  auto *const point_flags = owner._steiner_point_flags.begin();
  std::fill(owner._steiner_point_flags.begin(),
            owner._steiner_point_flags.end(), char(0));
  for (Index face = 0; face < n_faces; ++face) {
    if ((label_data[std::size_t(face)] & 1) == 0)
      continue;
    const auto &stored = triangle_data[std::size_t(face)];
    const std::array<Index, 3> triangle = {stored.v[0], stored.v[1],
                                           stored.v[2]};
    const std::array<bool, 3> constrained = {
        stored.seg[0] != Owner::none, stored.seg[1] != Owner::none,
        stored.seg[2] != Owner::none};
    for (int edge = 0; edge < 3; ++edge) {
      point_flags[std::size_t(triangle[std::size_t(edge)])] |= used;
      if (constrained[std::size_t(edge)]) {
        point_flags[std::size_t(triangle[std::size_t(edge)])] |= on_constraint;
        point_flags[std::size_t(triangle[std::size_t((edge + 1) % 3)])] |=
            on_constraint;
      }
    }
  }

  for (Index input = 0; input < n_input_points; ++input) {
    const auto output = input_map_data[std::size_t(input)];
    if (std::make_signed_t<Index>(output) >= 0 &&
        output < Index(n_points))
      point_flags[std::size_t(output)] |= is_input;
  }

  for (std::size_t point = 0; point < n_points; ++point)
    if (point_flags[point] == used)
      f(point_data[point]);
}

} // namespace tf::topology::cdt
