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

#include "trueform/cpp/geometry/sharp_edges.hpp"

#include "trueform/geometry/make_sharp_edges.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto sharp_edges(const mesh<Index, Real, Dims, Ngon> &value,
                 tf::rad<Real> angle_threshold) -> nd_array<Index> {
  value.require_indices();
  auto edges = tf::make_sharp_edges(
      value.polygons() | tf::tag(value.manifold_edge_link()), angle_threshold);
  const auto number_of_edges = static_cast<int>(edges.size());
  return nd_array<Index>::from_buffer(std::move(edges.data_buffer()),
                                      {number_of_edges, 2});
}

} // namespace tf::cpp
