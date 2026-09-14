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

#include "nd_array.hpp"

#include "trueform/core/curves_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf::cpp::test {

enum class orientation_mode { preserve, ignore };

template <typename T> using edge_signature = std::array<T, 2>;
template <typename Real> using point_signature = std::array<Real, 3>;
template <typename Real>
using path_signature = std::vector<point_signature<Real>>;
using point_id_path_signature = std::vector<std::int32_t>;

template <typename Point> struct curve_path_signature {
  bool closed = false;
  std::vector<Point> points;

  friend auto operator==(const curve_path_signature &left,
                         const curve_path_signature &right) -> bool {
    return left.closed == right.closed && left.points == right.points;
  }
  friend auto operator!=(const curve_path_signature &left,
                         const curve_path_signature &right) -> bool {
    return !(left == right);
  }
  friend auto operator<(const curve_path_signature &left,
                        const curve_path_signature &right) -> bool {
    if (left.closed != right.closed)
      return left.closed < right.closed;
    return left.points < right.points;
  }
};

template <typename Real>
using geometry_curve_path_signature =
    curve_path_signature<point_signature<Real>>;
using topology_curve_path_signature = curve_path_signature<std::int32_t>;

template <typename Real> struct mesh_geometry_signature {
  std::vector<point_signature<Real>> point_multiset;
  std::vector<path_signature<Real>> faces;

  friend auto operator==(const mesh_geometry_signature &left,
                         const mesh_geometry_signature &right) -> bool {
    return left.point_multiset == right.point_multiset &&
           left.faces == right.faces;
  }
  friend auto operator!=(const mesh_geometry_signature &left,
                         const mesh_geometry_signature &right) -> bool {
    return !(left == right);
  }
};

template <typename Real> struct mesh_topology_signature {
  std::vector<point_signature<Real>> point_storage;
  std::vector<point_id_path_signature> faces;

  friend auto operator==(const mesh_topology_signature &left,
                         const mesh_topology_signature &right) -> bool {
    return left.point_storage == right.point_storage &&
           left.faces == right.faces;
  }
  friend auto operator!=(const mesh_topology_signature &left,
                         const mesh_topology_signature &right) -> bool {
    return !(left == right);
  }
};

template <typename Real> struct curves_geometry_signature {
  std::vector<point_signature<Real>> point_multiset;
  std::vector<geometry_curve_path_signature<Real>> paths;

  friend auto operator==(const curves_geometry_signature &left,
                         const curves_geometry_signature &right) -> bool {
    return left.point_multiset == right.point_multiset &&
           left.paths == right.paths;
  }
  friend auto operator!=(const curves_geometry_signature &left,
                         const curves_geometry_signature &right) -> bool {
    return !(left == right);
  }
};

template <typename Real> struct curves_topology_signature {
  std::vector<point_signature<Real>> point_storage;
  std::vector<topology_curve_path_signature> paths;

  friend auto operator==(const curves_topology_signature &left,
                         const curves_topology_signature &right) -> bool {
    return left.point_storage == right.point_storage &&
           left.paths == right.paths;
  }
  friend auto operator!=(const curves_topology_signature &left,
                         const curves_topology_signature &right) -> bool {
    return !(left == right);
  }
};

namespace detail {

template <typename T>
auto minimum_rotation(const std::vector<T> &values) -> std::vector<T> {
  if (values.empty())
    return {};

  auto best = values;
  for (std::size_t shift = 1; shift < values.size(); ++shift) {
    std::vector<T> candidate;
    candidate.reserve(values.size());
    candidate.insert(candidate.end(), values.begin() + shift, values.end());
    candidate.insert(candidate.end(), values.begin(), values.begin() + shift);
    if (candidate < best)
      best = std::move(candidate);
  }
  return best;
}

template <typename T>
auto require_vector_shape(const tf::cpp::nd_array<T> &values,
                          const char *message) -> void {
  if (values.ndim() != 1 || values.shape_at(0) < 0 ||
      static_cast<std::size_t>(values.shape_at(0)) != values.length())
    throw std::invalid_argument(message);
}

template <typename T>
auto require_three_column_shape(const tf::cpp::nd_array<T> &values,
                                const char *message) -> void {
  if (values.ndim() != 2 || values.shape_at(0) < 0 || values.shape_at(1) != 3 ||
      static_cast<std::size_t>(values.shape_at(0)) != values.length() / 3 ||
      values.length() % 3 != 0)
    throw std::invalid_argument(message);
}

template <typename Real>
auto point_at(const tf::cpp::nd_array<Real> &points, std::int32_t id)
    -> point_signature<Real> {
  if (id < 0 || static_cast<std::size_t>(id) >= points.length() / 3)
    throw std::out_of_range("canonical point index is out of range");
  const auto offset = static_cast<std::size_t>(id) * 3;
  return {points[offset], points[offset + 1], points[offset + 2]};
}

template <typename Real>
auto point_storage(const tf::cpp::nd_array<Real> &points)
    -> std::vector<point_signature<Real>> {
  std::vector<point_signature<Real>> result;
  result.reserve(points.length() / 3);
  for (std::size_t point = 0; point < points.length() / 3; ++point)
    result.push_back(point_at(points, static_cast<std::int32_t>(point)));
  return result;
}

template <typename Real> struct validated_mesh {
  tf::cpp::nd_array<std::int32_t> faces;
  tf::cpp::nd_array<Real> points;
};

template <typename Real>
auto validate_mesh(
    const tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3> &value,
    bool require_no_orphaned_points) -> validated_mesh<Real> {
  auto faces =
      copied_nd_array(value.faces_buffer().data_buffer(),
                      {static_cast<int>(value.faces_buffer().size()), 3});
  auto points =
      copied_nd_array(value.points_buffer().data_buffer(),
                      {static_cast<int>(value.points_buffer().size()), 3});
  require_three_column_shape(faces, "mesh faces must have shape (n, 3)");
  require_three_column_shape(points, "mesh points must have shape (n, 3)");

  std::vector<bool> referenced(points.length() / 3, false);
  for (const auto id : faces) {
    if (id < 0 || static_cast<std::size_t>(id) >= referenced.size())
      throw std::out_of_range("mesh face point index is out of range");
    referenced[static_cast<std::size_t>(id)] = true;
  }
  if (require_no_orphaned_points &&
      std::find(referenced.begin(), referenced.end(), false) !=
          referenced.end())
    throw std::invalid_argument("mesh contains orphaned points");

  return {std::move(faces), std::move(points)};
}

template <typename Real> struct validated_curves {
  tf::cpp::nd_array<std::int32_t> offsets;
  tf::cpp::nd_array<std::int32_t> ids;
  tf::cpp::nd_array<Real> points;
};

template <typename Real>
auto validate_curves(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &value,
    bool require_no_orphaned_points) -> validated_curves<Real> {
  auto arrays = tf::cpp::test::arrays_of(value);
  auto offsets = std::move(arrays.offsets);
  auto ids = std::move(arrays.ids);
  auto points = std::move(arrays.points);
  require_vector_shape(offsets, "curve offsets must have shape (n)");
  require_vector_shape(ids, "curve packed point IDs must have shape (n)");
  require_three_column_shape(points, "curve points must have shape (n, 3)");

  if (offsets.empty())
    throw std::invalid_argument("curve offsets must be nonempty");
  if (offsets[0] != 0)
    throw std::invalid_argument("curve offsets must start at zero");
  for (std::size_t offset = 1; offset < offsets.length(); ++offset)
    if (offsets[offset] < offsets[offset - 1])
      throw std::invalid_argument("curve offsets must be nondecreasing");
  if (offsets[offsets.length() - 1] < 0 ||
      static_cast<std::size_t>(offsets[offsets.length() - 1]) != ids.length())
    throw std::invalid_argument(
        "curve final offset must equal packed point ID count");

  std::vector<bool> referenced(points.length() / 3, false);
  for (const auto id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= referenced.size())
      throw std::out_of_range("curve path point index is out of range");
    referenced[static_cast<std::size_t>(id)] = true;
  }
  if (require_no_orphaned_points &&
      std::find(referenced.begin(), referenced.end(), false) !=
          referenced.end())
    throw std::invalid_argument("curves contain orphaned points");

  return {std::move(offsets), std::move(ids), std::move(points)};
}

} // namespace detail

template <typename T>
auto canonicalize_oriented_edges(const tf::cpp::nd_array<T> &edges)
    -> std::vector<edge_signature<T>> {
  if (edges.ndim() != 2 || edges.shape_at(0) < 0 || edges.shape_at(1) != 2 ||
      static_cast<std::size_t>(edges.shape_at(0)) != edges.length() / 2 ||
      edges.length() % 2 != 0)
    throw std::invalid_argument("edge array must have shape (n, 2)");

  std::vector<edge_signature<T>> result;
  result.reserve(static_cast<std::size_t>(edges.shape_at(0)));
  for (std::size_t offset = 0; offset < edges.length(); offset += 2)
    result.push_back({edges[offset], edges[offset + 1]});
  std::sort(result.begin(), result.end());
  return result;
}

template <typename T>
auto canonicalize_unoriented_edges(const tf::cpp::nd_array<T> &edges)
    -> std::vector<edge_signature<T>> {
  auto result = canonicalize_oriented_edges(edges);
  for (auto &edge : result)
    if (edge[1] < edge[0])
      std::swap(edge[0], edge[1]);
  std::sort(result.begin(), result.end());
  return result;
}

template <typename T>
auto canonicalize_cyclic_face(
    std::vector<T> face, orientation_mode mode = orientation_mode::preserve)
    -> std::vector<T> {
  auto result = detail::minimum_rotation(face);
  if (mode == orientation_mode::ignore) {
    std::reverse(face.begin(), face.end());
    result = std::min(result, detail::minimum_rotation(face));
  }
  return result;
}

template <typename T>
auto canonicalize_open_path(std::vector<T> path,
                            orientation_mode mode = orientation_mode::preserve)
    -> std::vector<T> {
  if (mode == orientation_mode::ignore) {
    auto reversed = path;
    std::reverse(reversed.begin(), reversed.end());
    path = std::min(path, reversed);
  }
  return path;
}

/// Canonicalize a closed path without retaining a duplicate closing vertex.
template <typename T>
auto canonicalize_closed_path(
    std::vector<T> path, orientation_mode mode = orientation_mode::preserve)
    -> std::vector<T> {
  if (path.size() > 1 && path.front() == path.back())
    path.pop_back();
  return canonicalize_cyclic_face(std::move(path), mode);
}

template <typename Real>
auto canonicalize_mesh_geometry(
    const tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3> &value,
    orientation_mode mode = orientation_mode::preserve,
    bool require_no_orphaned_points = false) -> mesh_geometry_signature<Real> {
  auto validated = detail::validate_mesh(value, require_no_orphaned_points);
  auto points = detail::point_storage(validated.points);
  std::sort(points.begin(), points.end());

  std::vector<path_signature<Real>> faces;
  faces.reserve(validated.faces.length() / 3);
  for (std::size_t offset = 0; offset < validated.faces.length(); offset += 3) {
    path_signature<Real> face{
        detail::point_at(validated.points, validated.faces[offset]),
        detail::point_at(validated.points, validated.faces[offset + 1]),
        detail::point_at(validated.points, validated.faces[offset + 2]),
    };
    faces.push_back(canonicalize_cyclic_face(std::move(face), mode));
  }
  std::sort(faces.begin(), faces.end());
  return {std::move(points), std::move(faces)};
}

template <typename Real>
auto canonicalize_mesh_topology(
    const tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3> &value,
    orientation_mode mode = orientation_mode::preserve,
    bool require_no_orphaned_points = false) -> mesh_topology_signature<Real> {
  auto validated = detail::validate_mesh(value, require_no_orphaned_points);
  std::vector<point_id_path_signature> faces;
  faces.reserve(validated.faces.length() / 3);
  for (std::size_t offset = 0; offset < validated.faces.length(); offset += 3) {
    point_id_path_signature face{validated.faces[offset],
                                 validated.faces[offset + 1],
                                 validated.faces[offset + 2]};
    faces.push_back(canonicalize_cyclic_face(std::move(face), mode));
  }
  std::sort(faces.begin(), faces.end());
  return {detail::point_storage(validated.points), std::move(faces)};
}

template <typename Real>
auto canonicalize_curves_geometry(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &value,
    orientation_mode mode = orientation_mode::preserve,
    bool require_no_orphaned_points = false)
    -> curves_geometry_signature<Real> {
  auto validated = detail::validate_curves(value, require_no_orphaned_points);
  auto points = detail::point_storage(validated.points);
  std::sort(points.begin(), points.end());

  std::vector<geometry_curve_path_signature<Real>> paths;
  paths.reserve(validated.offsets.length() - 1);
  for (std::size_t path_id = 0; path_id + 1 < validated.offsets.length();
       ++path_id) {
    const auto begin = validated.offsets[path_id];
    const auto end = validated.offsets[path_id + 1];
    path_signature<Real> path;
    path.reserve(static_cast<std::size_t>(end - begin));
    for (auto index = begin; index < end; ++index)
      path.push_back(detail::point_at(
          validated.points, validated.ids[static_cast<std::size_t>(index)]));

    const auto closed = path.size() > 1 && path.front() == path.back();
    auto canonical = closed ? canonicalize_closed_path(std::move(path), mode)
                            : canonicalize_open_path(std::move(path), mode);
    paths.push_back({closed, std::move(canonical)});
  }
  std::sort(paths.begin(), paths.end());
  return {std::move(points), std::move(paths)};
}

template <typename Real>
auto canonicalize_curves_topology(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &value,
    orientation_mode mode = orientation_mode::preserve,
    bool require_no_orphaned_points = false)
    -> curves_topology_signature<Real> {
  auto validated = detail::validate_curves(value, require_no_orphaned_points);
  std::vector<topology_curve_path_signature> paths;
  paths.reserve(validated.offsets.length() - 1);
  for (std::size_t path_id = 0; path_id + 1 < validated.offsets.length();
       ++path_id) {
    const auto begin = validated.offsets[path_id];
    const auto end = validated.offsets[path_id + 1];
    point_id_path_signature path;
    path.reserve(static_cast<std::size_t>(end - begin));
    for (auto index = begin; index < end; ++index)
      path.push_back(validated.ids[static_cast<std::size_t>(index)]);

    const auto closed = path.size() > 1 && path.front() == path.back();
    auto canonical = closed ? canonicalize_closed_path(std::move(path), mode)
                            : canonicalize_open_path(std::move(path), mode);
    paths.push_back({closed, std::move(canonical)});
  }
  std::sort(paths.begin(), paths.end());
  return {detail::point_storage(validated.points), std::move(paths)};
}

} // namespace tf::cpp::test
