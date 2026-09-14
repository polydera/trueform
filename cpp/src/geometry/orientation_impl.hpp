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

#include "trueform/cpp/geometry/positively_oriented.hpp"
#include "trueform/cpp/geometry/reverse_winding.hpp"

#include "../core/materialized_mesh.hpp"

#include "trueform/core/polygons.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/geometry/ensure_positive_orientation.hpp"
#include "trueform/topology/reverse_winding.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {
namespace detail {

/// The winding is rewritten, so what is oriented is the caller's own copy of
/// the reading; the edge link is the source mesh's, because the copy has its
/// connectivity.
template <typename Index, typename Real, std::size_t Ngon>
auto positively_oriented_impl(const mesh<Index, Real, 3, Ngon> &value,
                              bool is_consistent)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  auto polygons = detail::materialized(value);
  if (polygons.size() != 0) {
    value.require_indices();
    auto oriented = polygons.polygons();
    if (is_consistent) {
      tf::ensure_positive_orientation(oriented, true);
    } else {
      auto tagged = oriented | tf::tag(value.manifold_edge_link());
      tf::ensure_positive_orientation(tagged, false);
    }
  }
  return polygons;
}

/// The winding is rewritten, so what is reversed is the caller's own copy of
/// the reading.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reverse_winding_impl(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  value.require_indices();
  auto polygons = detail::materialized(value);
  auto faces = polygons.faces();
  tf::reverse_winding(faces);
  return polygons;
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto positively_oriented(const mesh<Index, Real, Dims, Ngon> &value,
                         bool is_consistent)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return detail::positively_oriented_impl(value, is_consistent);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reverse_winding(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return detail::reverse_winding_impl(value);
}

} // namespace tf::cpp
