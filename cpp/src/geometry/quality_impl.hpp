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

#include "trueform/cpp/geometry/dihedral_angles.hpp"
#include "trueform/cpp/geometry/face_quality.hpp"

#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/angle.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/policy/normals.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/geometry/compute_dihedral_angles.hpp"
#include "trueform/geometry/compute_face_quality.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

/// An angle crosses the layer as the number it is, so the strong type is
/// stripped once, here, and the arrays the caller reads are radians.
template <typename Real>
auto radians_array(tf::buffer<tf::rad<Real>> &&angles, int count)
    -> nd_array<Real> {
  tf::buffer<Real> values;
  values.allocate(angles.size());
  tf::parallel_transform(
      angles, values, [](tf::rad<Real> angle) { return angle.value; },
      tf::checked);
  return nd_array<Real>::from_buffer(std::move(values), {count});
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto face_quality(const mesh<Index, Real, Dims, Ngon> &value)
    -> face_quality_result<Real> {
  value.require_indices();
  auto measured = tf::compute_face_quality(value.polygons());
  const auto count = static_cast<int>(value.number_of_faces());
  return {nd_array<Real>::from_buffer(std::move(measured.quality), {count}),
          detail::radians_array<Real>(std::move(measured.min_angle), count),
          detail::radians_array<Real>(std::move(measured.max_angle), count),
          nd_array<Real>::from_buffer(std::move(measured.aspect_ratio),
                                      {count})};
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto dihedral_angles(const mesh<Index, Real, Dims, Ngon> &value)
    -> dihedral_angles_result<Index, Real> {
  auto measured = tf::compute_dihedral_angles(
      value.polygons() | tf::tag(value.manifold_edge_link()) |
      tf::tag_normals(value.face_normals()));
  const auto count = static_cast<int>(measured.edges.size());
  return {nd_array<Index>::from_buffer(std::move(measured.edges.data_buffer()),
                                       {count, 2}),
          detail::radians_array<Real>(std::move(measured.angles), count)};
}

} // namespace tf::cpp
