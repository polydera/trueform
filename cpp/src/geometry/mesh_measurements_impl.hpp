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

#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/max_edge_length.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/min_edge_length.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/geometry/volume.hpp"

#include "trueform/core/area.hpp"
#include "trueform/core/max_edge_length.hpp"
#include "trueform/core/mean_edge_length.hpp"
#include "trueform/core/min_edge_length.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/core/signed_volume.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace tf::cpp {
namespace detail {

/// A mesh with no faces measures zero — no area, no length, no volume — and
/// that is one fact for the whole family, so it is stated here and nowhere
/// else.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Function>
auto measure_mesh(const mesh<Index, Real, Dims, Ngon> &value,
                  Function &&function) -> Real {
  value.require_indices();
  if (value.number_of_faces() == 0)
    return Real{0};
  return std::invoke(function, value.polygons() | tf::tag(value.frame()));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto mesh_signed_volume(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  static_assert(Dims == 3, "mesh_signed_volume requires a 3D mesh");
  return measure_mesh(value, [](const auto &polygons) -> Real {
    return tf::signed_volume(polygons);
  });
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto area(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return detail::measure_mesh(
      value, [](const auto &polygons) -> Real { return tf::area(polygons); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto mean_edge_length(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return detail::measure_mesh(value, [](const auto &polygons) -> Real {
    return tf::mean_edge_length(polygons);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto min_edge_length(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return detail::measure_mesh(value, [](const auto &polygons) -> Real {
    return tf::min_edge_length(polygons);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto max_edge_length(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return detail::measure_mesh(value, [](const auto &polygons) -> Real {
    return tf::max_edge_length(polygons);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto signed_volume(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return detail::mesh_signed_volume(value);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto volume(const mesh<Index, Real, Dims, Ngon> &value) -> Real {
  return std::abs(detail::mesh_signed_volume(value));
}

} // namespace tf::cpp
