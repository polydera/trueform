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
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

enum class delaunay_axis : std::uint8_t { x, y };

constexpr auto axis_index(delaunay_axis axis) -> std::size_t {
  return static_cast<std::size_t>(axis);
}

constexpr auto perpendicular(delaunay_axis axis) -> delaunay_axis {
  return axis == delaunay_axis::x ? delaunay_axis::y : delaunay_axis::x;
}

template <typename Index> struct delaunay_boundary {
  Index minimum;
  Index maximum;
};

template <typename Index> struct delaunay_frontier {
  auto operator[](delaunay_axis axis) -> delaunay_boundary<Index> & {
    return _axis[axis_index(axis)];
  }

  auto operator[](delaunay_axis axis) const
      -> const delaunay_boundary<Index> & {
    return _axis[axis_index(axis)];
  }

  std::array<delaunay_boundary<Index>, 2> _axis{};
};

template <typename Index> struct delaunay_build_result {
  Index outer_edge;
};

} // namespace tf::topology::cdt
