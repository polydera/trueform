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

#include "trueform/cpp/geometry/triangulate.hpp"

#include "../core/materialized_mesh.hpp"
#include "../core/require_point_indices.hpp"

#include "trueform/clean/polygons.hpp"
#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/geometry/triangulated.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tf::cpp {
namespace detail {

template <typename Index, typename Real, std::size_t Dims>
auto require_triangulate_axes() -> void {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "triangulate Real must be float or double");
  static_assert(is_supported_index_v<Index>,
                "triangulate Index must be int32 or int64");
  static_assert(Dims == 2 || Dims == 3,
                "triangulate Dims must be two or three");
}

template <typename Real, std::size_t Dims>
auto require_triangulate_points(const nd_array<Real> &points) -> void {
  if (!points.is_valid() || points.ndim() != 2 ||
      points.shape_at(1) != static_cast<int>(Dims))
    throw std::invalid_argument(
        Dims == 2 ? "triangulate: points must have shape [N, 2]"
                  : "triangulate: points must have shape [N, 3]");
}

template <typename Index>
auto require_fixed_faces(const nd_array<Index> &faces) -> int {
  if (!faces.is_valid() || faces.ndim() != 2 || faces.shape_at(1) < 3)
    throw std::invalid_argument(
        "triangulate: fixed faces must have shape [N, V] with V >= 3");
  return faces.shape_at(1);
}

template <typename Index>
auto require_dynamic_faces(const offset_blocked_buffer<Index, Index> &faces)
    -> void {
  if (!faces.is_valid())
    throw std::invalid_argument("triangulate: dynamic faces must be valid");
  const auto offsets = faces.offsets();
  const auto data = faces.data();
  if (offsets.ndim() != 1 || data.ndim() != 1)
    throw std::invalid_argument(
        "triangulate: dynamic face storage must be one-dimensional");
  if (offsets.empty()) {
    if (!data.empty())
      throw std::invalid_argument(
          "triangulate: empty offsets require empty face data");
    return;
  }
  if (offsets[0] != Index{0})
    throw std::invalid_argument(
        "triangulate: first dynamic face offset must be zero");
  for (std::size_t index = 1; index < offsets.length(); ++index) {
    if (offsets[index] < offsets[index - 1])
      throw std::invalid_argument(
          "triangulate: dynamic face offsets must be nondecreasing");
    if (offsets[index] - offsets[index - 1] < Index{3})
      throw std::invalid_argument(
          "triangulate: dynamic faces must contain at least three indices");
  }
  if (offsets[offsets.length() - 1] != static_cast<Index>(data.length()))
    throw std::invalid_argument(
        "triangulate: final dynamic face offset must equal data size");
}

/// The caller's points in core's own storage, stating no face: what an input
/// that triangulates to nothing answers, and what a width-three input writes
/// its own faces beside.
template <typename Index, typename Real, std::size_t Dims>
auto mesh_over_points(const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  tf::polygons_buffer<Index, Real, Dims, 3> output;
  output.points_buffer().allocate(points.length() / Dims);
  tf::parallel_copy(points.make_range(), output.points_buffer().data_buffer());
  return output;
}

/// Triangulation reads the mesh's raw stored geometry; its frame is
/// intentionally neither applied nor transferred, and a mesh already stored as
/// triangles is its own answer. A resolved face mints, so the resolved
/// triangulation owns its own point table — the input's vertices, then the
/// mints — and the caller's points are not carried through.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto triangulate_mesh(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  require_triangulate_axes<Index, Real, Dims>();
  value.require_indices();
  if constexpr (Ngon == 3)
    return detail::materialized(value);
  else
    return tf::triangulated<Index>(value.polygons());
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate_dynamic(const offset_blocked_buffer<Index, Index> &faces,
                         const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  require_triangulate_axes<Index, Real, Dims>();
  require_dynamic_faces(faces);
  require_triangulate_points<Real, Dims>(points);
  const auto face_indices = faces.data();
  carrier::require_point_indices(face_indices.make_range(), points.shape_at(0),
                                 "triangulate");
  if (faces.size() == 0)
    return mesh_over_points<Index, Real, Dims>(points);

  auto polygons = tf::make_polygons(tf::make_faces(faces.make_range()),
                                    tf::make_points<Dims>(points.make_range()));
  return tf::triangulated<Index>(polygons);
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate_fixed(const nd_array<Index> &faces,
                       const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  require_triangulate_axes<Index, Real, Dims>();
  const auto face_size = require_fixed_faces(faces);
  require_triangulate_points<Real, Dims>(points);
  carrier::require_point_indices(faces.make_range(), points.shape_at(0),
                                 "triangulate");

  if (faces.shape_at(0) == 0)
    return mesh_over_points<Index, Real, Dims>(points);
  if (face_size == 3) {
    auto output = mesh_over_points<Index, Real, Dims>(points);
    output.faces_buffer().allocate(static_cast<std::size_t>(faces.shape_at(0)));
    tf::parallel_copy(faces.make_range(), output.faces_buffer().data_buffer());
    return output;
  }

  auto polygons = tf::make_polygons(
      tf::make_faces(tf::make_blocked_range(faces.make_range(), face_size)),
      tf::make_points<Dims>(points.make_range()));
  return tf::triangulated<Index>(polygons);
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate_polygon_array(const nd_array<Real> &polygons)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  require_triangulate_axes<Index, Real, Dims>();
  if (!polygons.is_valid())
    throw std::invalid_argument("triangulate: polygons must be valid");

  if (polygons.ndim() == 2) {
    if (polygons.shape_at(1) != static_cast<int>(Dims) ||
        polygons.shape_at(0) < 3)
      throw std::invalid_argument(
          "triangulate: polygon must have shape [V, Dims] with V >= 3");
    auto points = tf::make_points<Dims>(polygons.make_range());
    return tf::triangulated<Index>(tf::make_polygon(points));
  }

  if (polygons.ndim() != 3 || polygons.shape_at(2) != static_cast<int>(Dims) ||
      polygons.shape_at(1) < 3)
    throw std::invalid_argument(
        "triangulate: polygon batch must have shape [N, V, Dims] with V >= 3");
  if (polygons.shape_at(0) == 0)
    return {};

  const auto polygon_size = polygons.shape_at(1);
  auto points = tf::make_points<Dims>(polygons.make_range());
  auto polygon_range = tf::make_polygons(tf::make_mapped_range(
      tf::make_blocked_range(points, polygon_size),
      [](const auto &block) { return tf::make_polygon(block); }));
  auto indexed = tf::cleaned<Index>(polygon_range);
  return tf::triangulated(indexed.polygons());
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto triangulate(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  return detail::triangulate_mesh(value);
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const offset_blocked_buffer<Index, Index> &faces,
                 const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  return detail::triangulate_dynamic<Index, Real, Dims>(faces, points);
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const nd_array<Index> &faces, const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  return detail::triangulate_fixed<Index, Real, Dims>(faces, points);
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const nd_array<Real> &polygons)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  return detail::triangulate_polygon_array<Index, Real, Dims>(polygons);
}

} // namespace tf::cpp
