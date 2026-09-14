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

#include "trueform/cpp/geometry/principal_curvatures.hpp"
#include "trueform/cpp/geometry/principal_directions.hpp"
#include "trueform/cpp/geometry/shape_index.hpp"

#include "trueform/core/policy/normals.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/geometry/compute_principal_curvatures.hpp"
#include "trueform/geometry/compute_shape_index.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/policy/vertex_link.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

template <typename Index, typename Real, std::size_t Ngon, typename Fn>
auto with_curvature_polygons(const mesh<Index, Real, 3, Ngon> &value, int k,
                             Fn &&fn) -> decltype(auto) {
  if (k < 0)
    throw std::invalid_argument("curvature: k must be non-negative");

  const auto number_of_points = static_cast<int>(value.number_of_points());

  auto points = value.points() | tf::tag_normals(value.point_normals());
  auto polygons = tf::make_polygons(value.faces(), points) |
                  tf::tag(value.face_membership()) |
                  tf::tag(value.vertex_link());
  return fn(polygons, number_of_points, static_cast<std::size_t>(k));
}

template <typename Index, typename Real, std::size_t Ngon>
auto principal_curvatures_impl(const mesh<Index, Real, 3, Ngon> &value, int k)
    -> principal_curvatures_result<Real> {
  return with_curvature_polygons(
      value, k,
      [](const auto &polygons, int number_of_points, std::size_t ring) {
        auto [k0, k1] = tf::make_principal_curvatures(polygons, ring);
        return principal_curvatures_result<Real>{
            nd_array<Real>::from_buffer(std::move(k0), {number_of_points}),
            nd_array<Real>::from_buffer(std::move(k1), {number_of_points}),
        };
      });
}

template <typename Index, typename Real, std::size_t Ngon>
auto principal_directions_impl(const mesh<Index, Real, 3, Ngon> &value, int k)
    -> principal_directions_result<Real> {
  return with_curvature_polygons(
      value, k,
      [](const auto &polygons, int number_of_points, std::size_t ring) {
        auto [k0, k1, d0, d1] = tf::make_principal_directions(polygons, ring);
        return principal_directions_result<Real>{
            nd_array<Real>::from_buffer(std::move(k0), {number_of_points}),
            nd_array<Real>::from_buffer(std::move(k1), {number_of_points}),
            nd_array<Real>::from_buffer(std::move(d0.data_buffer()),
                                        {number_of_points, 3}),
            nd_array<Real>::from_buffer(std::move(d1.data_buffer()),
                                        {number_of_points, 3}),
        };
      });
}

template <typename Index, typename Real, std::size_t Ngon>
auto shape_index_impl(const mesh<Index, Real, 3, Ngon> &value, int k)
    -> nd_array<Real> {
  return with_curvature_polygons(
      value, k,
      [](const auto &polygons, int number_of_points, std::size_t ring) {
        auto result = tf::make_shape_index(polygons, ring);
        return nd_array<Real>::from_buffer(std::move(result),
                                           {number_of_points});
      });
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto principal_curvatures(const mesh<Index, Real, Dims, Ngon> &value, int k)
    -> principal_curvatures_result<Real> {
  return detail::principal_curvatures_impl(value, k);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto principal_directions(const mesh<Index, Real, Dims, Ngon> &value, int k)
    -> principal_directions_result<Real> {
  return detail::principal_directions_impl(value, k);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto shape_index(const mesh<Index, Real, Dims, Ngon> &value, int k)
    -> nd_array<Real> {
  return detail::shape_index_impl(value, k);
}

} // namespace tf::cpp
