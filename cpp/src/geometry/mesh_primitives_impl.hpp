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

#include "../core/carrier_arrays.hpp"

#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_cylinder_mesh.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/make_tube_mesh.hpp"

#include "trueform/core/dot.hpp"
#include "trueform/core/epsilon.hpp"
#include "trueform/geometry/make_box_mesh.hpp"
#include "trueform/geometry/make_cylinder_mesh.hpp"
#include "trueform/geometry/make_plane_mesh.hpp"
#include "trueform/geometry/make_sphere_mesh.hpp"
#include "trueform/geometry/make_tube_mesh.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace tf::cpp {
namespace detail {

inline auto require_minimum(const char *name, std::int32_t value,
                            std::int32_t minimum) -> void {
  if (value < minimum)
    throw std::invalid_argument(std::string(name) + " must be at least " +
                                std::to_string(minimum));
}

inline auto require_index_capacity(const char *name, std::int64_t value)
    -> void {
  if (value > std::numeric_limits<std::int32_t>::max())
    throw std::length_error(std::string(name) + " exceeds int32 capacity");
}

inline auto require_scaled_index_capacity(const char *name, std::int64_t value,
                                          std::int32_t scale) -> void {
  if (value > std::numeric_limits<std::int32_t>::max() / scale)
    throw std::length_error(std::string(name) + " exceeds int32 capacity");
}

inline auto require_sphere_counts(std::int32_t stacks, std::int32_t segments)
    -> void {
  require_minimum("sphere stacks", stacks, 2);
  require_minimum("sphere segments", segments, 3);
  const auto rings = static_cast<std::int64_t>(stacks) - 1;
  const auto number_of_points = 2 + rings * static_cast<std::int64_t>(segments);
  const auto number_of_faces = 2 * rings * static_cast<std::int64_t>(segments);
  require_scaled_index_capacity("sphere point storage", number_of_points, 3);
  require_scaled_index_capacity("sphere face storage", number_of_faces, 3);
}

inline auto require_cylinder_count(std::int32_t segments) -> void {
  require_minimum("cylinder segments", segments, 3);
  const auto number_of_points = 2 + 2 * static_cast<std::int64_t>(segments);
  const auto number_of_faces = 4 * static_cast<std::int64_t>(segments);
  require_scaled_index_capacity("cylinder point storage", number_of_points, 3);
  require_scaled_index_capacity("cylinder face storage", number_of_faces, 3);
}

inline auto require_box_counts(std::int32_t width_ticks,
                               std::int32_t height_ticks,
                               std::int32_t depth_ticks) -> void {
  require_minimum("box width ticks", width_ticks, 1);
  require_minimum("box height ticks", height_ticks, 1);
  require_minimum("box depth ticks", depth_ticks, 1);
  const auto width = static_cast<std::uint64_t>(width_ticks);
  const auto height = static_cast<std::uint64_t>(height_ticks);
  const auto depth = static_cast<std::uint64_t>(depth_ticks);
  const auto number_of_face_cells =
      width * height + depth * height + width * depth;
  if (number_of_face_cells >
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) / 36)
    throw std::length_error(
        "subdivided box intermediate storage exceeds int32 capacity");
  require_index_capacity("subdivided box face count",
                         static_cast<std::int64_t>(4 * number_of_face_cells));
}

inline auto require_plane_counts(std::int32_t width_ticks,
                                 std::int32_t height_ticks) -> void {
  require_minimum("plane width ticks", width_ticks, 1);
  require_minimum("plane height ticks", height_ticks, 1);
  const auto width = static_cast<std::int64_t>(width_ticks);
  const auto height = static_cast<std::int64_t>(height_ticks);
  const auto number_of_points = (width + 1) * (height + 1);
  const auto number_of_faces = 2 * width * height;
  require_scaled_index_capacity("plane point storage", number_of_points, 3);
  require_scaled_index_capacity("plane face storage", number_of_faces, 3);
}

/// The two arrays a polyline is, read as the curves they name: the paths
/// state the blocks and every id in them names a point this table has, which
/// is the carrier door every array-taking entry asks.
template <typename Index, typename Real>
auto tube_curves(const offset_blocked_buffer<Index, Index> &paths,
                 const nd_array<Real> &points) {
  carrier::require_columns<3>(points, "tube points");
  static_cast<void>(carrier::require_offset_blocks(
      paths, carrier::any_count, points.shape_at(0), "tube paths"));
  return tf::make_curves(tf::make_paths(paths.make_range()),
                         tf::make_points<3>(points.make_range()));
}

template <typename Curves>
auto require_tube_input(const Curves &source, std::int32_t radial_segments)
    -> void {
  using Real = tf::coordinate_type<Curves>;
  require_minimum("tube radial segments", radial_segments, 3);

  std::int64_t number_of_rings = 0;
  std::int64_t number_of_connections = 0;
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto curve = source[index];
    const auto number_of_points = static_cast<std::size_t>(curve.size());
    if (number_of_points < 2)
      continue;
    const auto difference = curve[number_of_points - 1] - curve[0];
    const auto closed =
        tf::dot(difference, difference) < tf::epsilon<Real> * tf::epsilon<Real>;
    const auto rings = closed ? number_of_points - 1 : number_of_points;
    number_of_rings += static_cast<std::int64_t>(rings);
    number_of_connections +=
        static_cast<std::int64_t>(closed ? rings : rings - 1);
  }

  const auto radial = static_cast<std::int64_t>(radial_segments);
  require_scaled_index_capacity("tube point storage", number_of_rings * radial,
                                3);
  require_scaled_index_capacity("tube face storage",
                                number_of_connections * radial, 6);
}

template <typename Index, typename Real>
auto make_sphere_mesh_impl(Real radius, std::int32_t stacks,
                           std::int32_t segments)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  require_sphere_counts(stacks, segments);
  return tf::make_sphere_mesh<Index>(radius, static_cast<Index>(stacks),
                                     static_cast<Index>(segments));
}

template <typename Index, typename Real>
auto make_cylinder_mesh_impl(Real radius, Real height, std::int32_t segments)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  require_cylinder_count(segments);
  return tf::make_cylinder_mesh<Index>(radius, height,
                                       static_cast<Index>(segments));
}

template <typename Index, typename Real>
auto make_box_mesh_impl(Real width, Real height, Real depth)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  return tf::make_box_mesh<Index>(width, height, depth);
}

template <typename Index, typename Real>
auto make_box_mesh_impl(Real width, Real height, Real depth,
                        std::int32_t width_ticks, std::int32_t height_ticks,
                        std::int32_t depth_ticks)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  require_box_counts(width_ticks, height_ticks, depth_ticks);
  return tf::make_box_mesh<Index>(
      width, height, depth, static_cast<Index>(width_ticks),
      static_cast<Index>(height_ticks), static_cast<Index>(depth_ticks));
}

template <typename Index, typename Real>
auto make_plane_mesh_impl(Real width, Real height, std::int32_t width_ticks,
                          std::int32_t height_ticks)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  require_plane_counts(width_ticks, height_ticks);
  return tf::make_plane_mesh<Index>(width, height,
                                    static_cast<Index>(width_ticks),
                                    static_cast<Index>(height_ticks));
}

} // namespace detail

template <typename Real>
auto make_sphere_mesh(Real radius, std::int32_t stacks, std::int32_t segments)
    -> tf::polygons_buffer<default_index_t, Real, 3, 3> {
  return detail::make_sphere_mesh_impl<std::int32_t, Real>(radius, stacks,
                                                           segments);
}

template <typename Index, typename Real>
auto make_sphere_mesh(Real radius, std::int32_t stacks, std::int32_t segments)
    -> typename detail::minted_mesh_result<Index, Real, 3>::type {
  return detail::make_sphere_mesh_impl<Index, Real>(radius, stacks, segments);
}

template <typename Real>
auto make_cylinder_mesh(Real radius, Real height, std::int32_t segments)
    -> tf::polygons_buffer<default_index_t, Real, 3, 3> {
  return detail::make_cylinder_mesh_impl<std::int32_t, Real>(radius, height,
                                                             segments);
}

template <typename Index, typename Real>
auto make_cylinder_mesh(Real radius, Real height, std::int32_t segments) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type {
  return detail::make_cylinder_mesh_impl<Index, Real>(radius, height, segments);
}

template <typename Real>
auto make_box_mesh(Real width, Real height, Real depth)
    -> tf::polygons_buffer<default_index_t, Real, 3, 3> {
  return detail::make_box_mesh_impl<std::int32_t, Real>(width, height, depth);
}

template <typename Index, typename Real>
auto make_box_mesh(Real width, Real height, Real depth) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type {
  return detail::make_box_mesh_impl<Index, Real>(width, height, depth);
}

template <typename Real>
auto make_box_mesh(Real width, Real height, Real depth,
                   std::int32_t width_ticks, std::int32_t height_ticks,
                   std::int32_t depth_ticks)
    -> tf::polygons_buffer<default_index_t, Real, 3, 3> {
  return detail::make_box_mesh_impl<std::int32_t, Real>(
      width, height, depth, width_ticks, height_ticks, depth_ticks);
}

template <typename Index, typename Real>
auto make_box_mesh(Real width, Real height, Real depth,
                   std::int32_t width_ticks, std::int32_t height_ticks,
                   std::int32_t depth_ticks) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type {
  return detail::make_box_mesh_impl<Index, Real>(
      width, height, depth, width_ticks, height_ticks, depth_ticks);
}

template <typename Real>
auto make_plane_mesh(Real width, Real height, std::int32_t width_ticks,
                     std::int32_t height_ticks)
    -> tf::polygons_buffer<default_index_t, Real, 3, 3> {
  return detail::make_plane_mesh_impl<std::int32_t, Real>(
      width, height, width_ticks, height_ticks);
}

template <typename Index, typename Real>
auto make_plane_mesh(Real width, Real height, std::int32_t width_ticks,
                     std::int32_t height_ticks) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type {
  return detail::make_plane_mesh_impl<Index, Real>(width, height, width_ticks,
                                                   height_ticks);
}

template <typename Index, typename Real>
auto make_tube_mesh(const offset_blocked_buffer<Index, Index> &paths,
                    const nd_array<Real> &points, Real radius,
                    std::int32_t radial_segments) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type {
  const auto source = detail::tube_curves(paths, points);
  detail::require_tube_input(source, radial_segments);
  return detail::minted_mesh_result_t<Index, Real, 3>(
      tf::make_tube_mesh<Index>(source, radius, radial_segments));
}

} // namespace tf::cpp
