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

#include "trueform/cpp/geometry/dirichlet_field.hpp"
#include "trueform/cpp/geometry/geodesic_distances.hpp"

#include "trueform/core/buffer.hpp"
#include "trueform/core/none.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/geometry/make_dirichlet_field.hpp"
#include "trueform/geometry/make_geodesic_distances.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/policy/vertex_link.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace detail {

template <typename Index>
auto require_point_ids(const nd_array<Index> &ids, std::size_t n_points,
                       const char *operation, bool distinct) -> void {
  if (!ids.is_valid() || ids.ndim() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": ids must be one-dimensional");
  tf::buffer<char> seen;
  if (distinct) {
    seen.allocate(n_points);
    for (auto &mark : seen)
      mark = 0;
  }
  for (const auto id : ids) {
    if (id < Index{0} || static_cast<std::size_t>(id) >= n_points)
      throw std::out_of_range(std::string(operation) +
                              ": point id out of range");
    if (distinct) {
      if (seen[static_cast<std::size_t>(id)])
        throw std::invalid_argument(std::string(operation) +
                                    ": point ids must be distinct");
      seen[static_cast<std::size_t>(id)] = 1;
    }
  }
}

template <typename Index, typename Real, typename Weights>
auto dirichlet_field_impl(const mesh<Index, Real, 3, 3> &value,
                          const Weights &corner_weights,
                          const nd_array<Index> &fixed_ids,
                          const nd_array<double> &fixed_values,
                          tf::dirichlet_order order)
    -> dirichlet_field_result<Real> {
  const auto n_points = value.number_of_points();
  require_point_ids(fixed_ids, n_points, "dirichlet_field", true);
  if (!fixed_values.is_valid() || fixed_values.ndim() != 1 ||
      fixed_values.length() != fixed_ids.length())
    throw std::invalid_argument(
        "dirichlet_field: one fixed value per fixed id");
  for (const auto v : fixed_values)
    if (!std::isfinite(v))
      throw std::invalid_argument(
          "dirichlet_field: fixed values must be finite");
  const auto polygons = tf::make_polygons(value.faces(), value.points()) |
                        tf::tag(value.face_membership());
  auto solved =
      tf::make_dirichlet_field(polygons, corner_weights, fixed_ids.make_range(),
                               fixed_values.make_range(), order);
  return {nd_array<Real>::from_buffer(std::move(solved.field),
                                      {static_cast<int>(n_points)}),
          solved.status};
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto geodesic_distances(const mesh<Index, Real, Dims, Ngon> &value,
                        const nd_array<Index> &seeds, double radius)
    -> nd_array<Real> {
  const auto n_points = value.number_of_points();
  detail::require_point_ids(seeds, n_points, "geodesic_distances", false);
  if (std::isnan(radius) || radius < 0.0)
    throw std::invalid_argument(
        "geodesic_distances: radius must be non-negative and not NaN");
  const auto polygons = tf::make_polygons(value.faces(), value.points()) |
                        tf::tag(value.face_membership()) |
                        tf::tag(value.vertex_link());
  return nd_array<Real>::from_buffer(
      tf::make_geodesic_distances(polygons, seeds.make_range(), Real(radius)),
      {static_cast<int>(n_points)});
}

template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<Dims == 3, int>>
auto dirichlet_field(const mesh<Index, Real, Dims, 3> &value,
                     const nd_array<double> &corner_weights,
                     const nd_array<Index> &fixed_ids,
                     const nd_array<double> &fixed_values,
                     tf::dirichlet_order order)
    -> dirichlet_field_result<Real> {
  if (!corner_weights.is_valid() ||
      corner_weights.length() != 3 * value.number_of_faces())
    throw std::invalid_argument(
        "dirichlet_field: corner_weights must hold three weights per face");
  return detail::dirichlet_field_impl(value, corner_weights.make_range(),
                                      fixed_ids, fixed_values, order);
}

template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<Dims == 3, int>>
auto dirichlet_field(const mesh<Index, Real, Dims, 3> &value,
                     const nd_array<Index> &fixed_ids,
                     const nd_array<double> &fixed_values,
                     tf::dirichlet_order order)
    -> dirichlet_field_result<Real> {
  return detail::dirichlet_field_impl(value, tf::none, fixed_ids, fixed_values,
                                      order);
}

} // namespace tf::cpp
