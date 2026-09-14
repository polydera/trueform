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

#include "canonicalize.hpp"

#include "carriers.hpp"
#include "fixtures.hpp"
#include "nd_array.hpp"
#include "tolerances.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp::test {

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &first,
                const tf::cpp::nd_array<T> &second) -> bool {
  if (first.raw_shape() != second.raw_shape() ||
      first.length() != second.length())
    return false;
  for (std::size_t index = 0; index < first.length(); ++index)
    if (first[index] != second[index])
      return false;
  return true;
}

/// A RESULT is core's own storage, so a check reads it where it lies, and the
/// two readers below are the borrowed arrays every check indexes through. An
/// OPERAND is what a test holds — that storage, a cache and a placement — so
/// the fixtures at the foot of this file state one and the checks that read a
/// source read its own `polygons`.
template <typename Real>
using parity_mesh = tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3>;

template <typename Real>
using parity_operand =
    tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real>
auto parity_faces(const parity_mesh<Real> &mesh)
    -> tf::cpp::nd_array<tf::cpp::default_index_t> {
  const auto &storage = mesh.faces_buffer().data_buffer();
  const auto count = static_cast<int>(mesh.faces_buffer().size());
  return tf::cpp::nd_array<tf::cpp::default_index_t>::from_borrowed(
      {}, const_cast<tf::cpp::default_index_t *>(storage.data()),
      storage.size(), {count, 3});
}

template <typename Real>
auto parity_points(const parity_mesh<Real> &mesh) -> tf::cpp::nd_array<Real> {
  const auto &storage = mesh.points_buffer().data_buffer();
  const auto count = static_cast<int>(mesh.points_buffer().size());
  return tf::cpp::nd_array<Real>::from_borrowed(
      {}, const_cast<Real *>(storage.data()), storage.size(), {count, 3});
}

template <typename Real>
auto parity_face_count(const parity_mesh<Real> &mesh) -> int {
  return static_cast<int>(mesh.faces_buffer().size());
}

template <typename Real>
auto parity_point_count(const parity_mesh<Real> &mesh) -> int {
  return static_cast<int>(mesh.points_buffer().size());
}

template <typename Real>
auto mesh_indices_are_valid(const parity_mesh<Real> &mesh) -> bool {
  const auto faces = parity_faces(mesh);
  const auto points = parity_points(mesh);
  if (faces.ndim() != 2 || faces.shape_at(1) != 3 || points.ndim() != 2 ||
      points.shape_at(1) != 3)
    return false;
  for (const auto id : faces.make_range())
    if (id < 0 || id >= parity_point_count(mesh))
      return false;
  for (const auto coordinate : points.make_range())
    if (!std::isfinite(coordinate))
      return false;
  return true;
}

/// A curves result is aligned when its offsets frame its own ids, every id
/// names a point it holds, and every coordinate is finite. `require_nonempty`
/// additionally asks that it carry at least one path of at least two points.
template <typename Index, typename Real>
auto curves_are_aligned(const tf::curves_buffer<Index, Real, 3> &curves,
                        bool require_nonempty) -> bool {
  const auto arrays = tf::cpp::test::arrays_of(curves);
  const auto &offsets = arrays.offsets;
  const auto &ids = arrays.ids;
  const auto &points = arrays.points;
  if (points.ndim() != 2 || points.shape_at(1) != 3)
    return false;
  if (curves.size() == 0) {
    if (!offsets.empty() || !ids.empty())
      return false;
  } else if (offsets.length() != static_cast<std::size_t>(curves.size() + 1) ||
             offsets[0] != Index{0} ||
             offsets[offsets.length() - 1] !=
                 static_cast<Index>(ids.length())) {
    return false;
  }
  if (require_nonempty && (curves.size() == 0 || points.empty()))
    return false;

  for (std::size_t path = 1; path < offsets.length(); ++path) {
    if (offsets[path] < offsets[path - 1])
      return false;
    if (require_nonempty && offsets[path] - offsets[path - 1] < Index{2})
      return false;
  }
  for (const auto id : ids.make_range())
    if (id < Index{0} || id >= static_cast<Index>(points.shape_at(0)))
      return false;
  for (const auto coordinate : points.make_range())
    if (!std::isfinite(coordinate))
      return false;
  return true;
}

template <typename Real>
auto all_curve_paths_closed(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &curves)
    -> bool {
  const auto arrays = tf::cpp::test::arrays_of(curves);
  const auto &offsets = arrays.offsets;
  const auto &ids = arrays.ids;
  if (curves.size() == 0)
    return false;
  for (std::size_t path = 1; path < offsets.length(); ++path) {
    const auto begin = offsets[path - 1];
    const auto end = offsets[path];
    if (end - begin < 2 || ids[static_cast<std::size_t>(begin)] !=
                               ids[static_cast<std::size_t>(end - 1)])
      return false;
  }
  return true;
}

template <typename Real>
using point_signature = tf::cpp::test::point_signature<Real>;

template <typename Real>
auto point_at(const parity_mesh<Real> &mesh, std::int32_t point)
    -> point_signature<Real> {
  const auto points = parity_points(mesh);
  if (point < 0 || point >= parity_point_count(mesh))
    throw std::out_of_range("test point index is out of range");
  const auto offset = static_cast<std::size_t>(point) * 3;
  return {points[offset], points[offset + 1], points[offset + 2]};
}

/// An operand carries the frame it is read in, so a check that compares a
/// result against its source asks for the source placed.
template <typename Real>
auto point_at(const parity_operand<Real> &operand, std::int32_t point,
              bool apply_transformation) -> point_signature<Real> {
  auto result = point_at(operand.polygons, point);
  if (!apply_transformation || !operand.placement.placed)
    return result;

  const auto &matrix = operand.placement.values;
  const auto x = result[0];
  const auto y = result[1];
  const auto z = result[2];
  const auto w = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15];
  if (w == Real{0})
    throw std::invalid_argument("test transformation maps a point to infinity");
  return {(matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]) / w,
          (matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]) / w,
          (matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]) / w};
}

template <typename Real>
auto subtract(const point_signature<Real> &first,
              const point_signature<Real> &second) -> point_signature<Real> {
  return {first[0] - second[0], first[1] - second[1], first[2] - second[2]};
}

template <typename Real>
auto dot(const point_signature<Real> &first,
         const point_signature<Real> &second) -> Real {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

template <typename Real>
auto cross(const point_signature<Real> &first,
           const point_signature<Real> &second) -> point_signature<Real> {
  return {first[1] * second[2] - first[2] * second[1],
          first[2] * second[0] - first[0] * second[2],
          first[0] * second[1] - first[1] * second[0]};
}

template <typename Real>
auto barycentric_coordinates(
    const point_signature<Real> &point,
    const std::array<point_signature<Real>, 3> &triangle,
    std::array<Real, 3> &weights) -> bool {
  const auto ab = subtract(triangle[1], triangle[0]);
  const auto ac = subtract(triangle[2], triangle[0]);
  const auto ap = subtract(point, triangle[0]);
  const auto normal = cross(ab, ac);
  const auto normal_length = std::sqrt(dot(normal, normal));
  if (normal_length == Real{0})
    return false;

  Real coordinate_scale = Real{1};
  for (const auto &vertex : triangle)
    for (const auto coordinate : vertex)
      coordinate_scale = std::max(coordinate_scale, std::abs(coordinate));
  const auto tolerance =
      tf::cpp::test::base_tolerance<Real>().absolute * Real{64};
  if (std::abs(dot(ap, normal)) > tolerance * coordinate_scale * normal_length)
    return false;

  const auto d00 = dot(ab, ab);
  const auto d01 = dot(ab, ac);
  const auto d11 = dot(ac, ac);
  const auto d20 = dot(ap, ab);
  const auto d21 = dot(ap, ac);
  const auto denominator = d00 * d11 - d01 * d01;
  if (std::abs(denominator) <= tolerance * tolerance)
    return false;
  weights[1] = (d11 * d20 - d01 * d21) / denominator;
  weights[2] = (d00 * d21 - d01 * d20) / denominator;
  weights[0] = Real{1} - weights[1] - weights[2];
  return weights[0] >= -tolerance && weights[1] >= -tolerance &&
         weights[2] >= -tolerance;
}

template <typename Real>
auto source_triangle(const parity_operand<Real> &mesh, std::int32_t face,
                     bool apply_transformation)
    -> std::array<point_signature<Real>, 3> {
  if (face < 0 || face >= parity_face_count(mesh.polygons))
    throw std::out_of_range("test source face index is out of range");
  const auto faces = parity_faces(mesh.polygons);
  const auto offset = static_cast<std::size_t>(face) * 3;
  return {point_at(mesh, faces[offset], apply_transformation),
          point_at(mesh, faces[offset + 1], apply_transformation),
          point_at(mesh, faces[offset + 2], apply_transformation)};
}

template <typename Real>
auto output_face_matches_source_triangle(const parity_mesh<Real> &output,
                                         std::size_t output_face,
                                         const parity_operand<Real> &source,
                                         std::int32_t source_face,
                                         bool transform_source) -> bool {
  const auto triangle = source_triangle(source, source_face, transform_source);
  const auto faces = parity_faces(output);
  const auto offset = output_face * 3;
  for (std::size_t corner = 0; corner < 3; ++corner) {
    std::array<Real, 3> weights{};
    if (!barycentric_coordinates(point_at(output, faces[offset + corner]),
                                 triangle, weights))
      return false;
  }
  return true;
}

template <typename Real> struct face_provenance_record {
  tf::cpp::test::path_signature<Real> geometry;
  std::int32_t source_tag;
  std::int32_t source_face;

  friend auto operator==(const face_provenance_record &first,
                         const face_provenance_record &second) -> bool {
    return std::tie(first.geometry, first.source_tag, first.source_face) ==
           std::tie(second.geometry, second.source_tag, second.source_face);
  }
  friend auto operator<(const face_provenance_record &first,
                        const face_provenance_record &second) -> bool {
    return std::tie(first.geometry, first.source_tag, first.source_face) <
           std::tie(second.geometry, second.source_tag, second.source_face);
  }
};

template <typename Real, typename TagAt, typename FaceAt>
auto canonical_face_provenance(const parity_mesh<Real> &mesh, TagAt tag_at,
                               FaceAt face_at)
    -> std::vector<face_provenance_record<Real>> {
  const auto faces = parity_faces(mesh);
  std::vector<face_provenance_record<Real>> records;
  records.reserve(static_cast<std::size_t>(parity_face_count(mesh)));
  for (int face = 0; face < parity_face_count(mesh); ++face) {
    const auto offset = static_cast<std::size_t>(face) * 3;
    tf::cpp::test::path_signature<Real> geometry{
        point_at(mesh, faces[offset]), point_at(mesh, faces[offset + 1]),
        point_at(mesh, faces[offset + 2])};
    records.push_back(
        {tf::cpp::test::canonicalize_cyclic_face(
             std::move(geometry), tf::cpp::test::orientation_mode::ignore),
         static_cast<std::int32_t>(tag_at(face)),
         static_cast<std::int32_t>(face_at(face))});
  }
  std::sort(records.begin(), records.end());
  return records;
}

template <typename Real, typename TagAt, typename FaceAt>
auto carrier_provenance_matches_sources(
    const parity_mesh<Real> &output, std::size_t label_count,
    const std::vector<parity_operand<Real>> &sources, bool transform_sources,
    TagAt tag_at, FaceAt face_at) -> bool {
  if (label_count != static_cast<std::size_t>(parity_face_count(output)))
    return false;
  for (std::size_t face = 0; face < label_count; ++face) {
    const auto source_tag = static_cast<std::int32_t>(tag_at(face));
    const auto source_face = static_cast<std::int32_t>(face_at(face));
    if (source_tag < 0 ||
        static_cast<std::size_t>(source_tag) >= sources.size() ||
        source_face < 0 ||
        source_face >=
            parity_face_count(
                sources[static_cast<std::size_t>(source_tag)].polygons) ||
        !output_face_matches_source_triangle(
            output, face, sources[static_cast<std::size_t>(source_tag)],
            source_face, transform_sources))
      return false;
  }
  return true;
}

template <typename Result, typename Real>
auto boolean_provenance_is_valid(const Result &result,
                                 const parity_operand<Real> &first,
                                 const parity_operand<Real> &second) -> bool {
  if (result.labels.ndim() != 1 || result.face_labels.ndim() != 1 ||
      result.face_labels.length() != result.labels.length())
    return false;
  const std::vector<parity_operand<Real>> sources{first, second};
  return carrier_provenance_matches_sources(
      result.mesh, result.labels.length(), sources, true,
      [&](std::size_t face) { return result.labels[face]; },
      [&](std::size_t face) { return result.face_labels[face]; });
}

template <typename Result, typename Real>
auto isoband_provenance_is_valid(const Result &result,
                                 const parity_operand<Real> &source,
                                 const tf::cpp::nd_array<Real> &scalars,
                                 const tf::cpp::nd_array<Real> &cuts) -> bool {
  if (result.labels.ndim() != 1 || result.face_labels.ndim() != 1 ||
      result.labels.length() != result.face_labels.length() ||
      scalars.ndim() != 1 ||
      scalars.length() !=
          static_cast<std::size_t>(parity_point_count(source.polygons)) ||
      cuts.ndim() != 1)
    return false;

  const auto output_faces = parity_faces(result.mesh);
  const auto source_faces = parity_faces(source.polygons);
  for (std::size_t face = 0; face < result.labels.length(); ++face) {
    const auto source_face = result.face_labels[face];
    if (source_face < 0 || source_face >= parity_face_count(source.polygons) ||
        !output_face_matches_source_triangle(result.mesh, face, source,
                                             source_face, false))
      return false;

    point_signature<Real> centroid{};
    const auto output_offset = face * 3;
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const auto point =
          point_at(result.mesh, output_faces[output_offset + corner]);
      for (std::size_t axis = 0; axis < 3; ++axis)
        centroid[axis] += point[axis] / Real{3};
    }
    std::array<Real, 3> weights{};
    if (!barycentric_coordinates(
            centroid, source_triangle(source, source_face, false), weights))
      return false;

    Real scalar = Real{0};
    const auto source_offset = static_cast<std::size_t>(source_face) * 3;
    for (std::size_t corner = 0; corner < 3; ++corner)
      scalar += weights[corner] * scalars[static_cast<std::size_t>(
                                      source_faces[source_offset + corner])];
    std::int32_t expected_band = 0;
    for (const auto cut : cuts.make_range())
      if (scalar > cut)
        ++expected_band;
    if (result.labels[face] != expected_band)
      return false;
  }
  return true;
}

struct curve_path_statistics {
  std::size_t paths = 0;
  std::vector<std::int32_t> edge_counts;
  std::size_t closed = 0;
  std::size_t open = 0;
  std::size_t endpoint_points = 0;

  friend auto operator==(const curve_path_statistics &first,
                         const curve_path_statistics &second) -> bool {
    return std::tie(first.paths, first.edge_counts, first.closed, first.open,
                    first.endpoint_points) ==
           std::tie(second.paths, second.edge_counts, second.closed,
                    second.open, second.endpoint_points);
  }
};

template <typename Real>
auto curve_statistics(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &curves)
    -> curve_path_statistics {
  const auto arrays = tf::cpp::test::arrays_of(curves);
  const auto &offsets = arrays.offsets;
  const auto &ids = arrays.ids;
  curve_path_statistics result;
  result.paths = static_cast<std::size_t>(curves.size());
  std::set<std::int32_t> endpoints;
  for (std::size_t path = 0; path + 1 < offsets.length(); ++path) {
    const auto begin = offsets[path];
    const auto end = offsets[path + 1];
    result.edge_counts.push_back(end - begin - 1);
    if (end - begin >= 2 && ids[static_cast<std::size_t>(begin)] ==
                                ids[static_cast<std::size_t>(end - 1)]) {
      ++result.closed;
    } else {
      ++result.open;
      if (end - begin >= 2) {
        endpoints.insert(ids[static_cast<std::size_t>(begin)]);
        endpoints.insert(ids[static_cast<std::size_t>(end - 1)]);
      }
    }
  }
  std::sort(result.edge_counts.begin(), result.edge_counts.end());
  result.endpoint_points = endpoints.size();
  return result;
}

/// The operands the parity suites state, each core's own storage plus the
/// placement it is read in.
template <typename Real> auto horizontal_triangle() -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{0},
       Real{1}, Real{0}, Real{10}, Real{10}, Real{3}, Real{12}, Real{10},
       Real{3}, Real{10.5}, Real{12}, Real{3}});
  return result;
}

template <typename Real> auto vertical_triangle() -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8},
      {Real{7},   Real{-0.5}, Real{-1}, Real{7},   Real{-0.5}, Real{1},
       Real{7},   Real{0.75}, Real{0},  Real{27},  Real{10},   Real{-2},
       Real{27},  Real{12},   Real{1},  Real{27},  Real{9},    Real{2},
       Real{-13}, Real{-10},  Real{-1}, Real{-13}, Real{-8},   Real{2},
       Real{-13}, Real{-11},  Real{3}});
  result.place({Real{1}, Real{0}, Real{0}, Real{-7}, Real{0}, Real{1}, Real{0},
                Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0},
                Real{0}, Real{1}});
  return result;
}

template <typename Real> auto crossing_triangles() -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {Real{-1},  Real{-1},   Real{0},  Real{1},    Real{-1},   Real{0},
       Real{0},   Real{1},    Real{0},  Real{10},   Real{10},   Real{3},
       Real{12},  Real{10},   Real{3},  Real{10.5}, Real{12},   Real{3},
       Real{0},   Real{-0.5}, Real{-1}, Real{0},    Real{-0.5}, Real{1},
       Real{0},   Real{0.75}, Real{0},  Real{20},   Real{10},   Real{-2},
       Real{20},  Real{12},   Real{1},  Real{20},   Real{9},    Real{2},
       Real{-20}, Real{-10},  Real{-1}, Real{-20},  Real{-8},   Real{2},
       Real{-20}, Real{-11},  Real{3}});
  return result;
}

template <typename Real>
auto three_way_triangles() -> std::vector<parity_operand<Real>> {
  std::vector<parity_operand<Real>> result;
  result.emplace_back();
  result.back().polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
          {0, 1, 2}, {Real{-3}, Real{-2}, Real{0}, Real{3}, Real{-2}, Real{0},
                      Real{0}, Real{3}, Real{0}});
  result.emplace_back();
  result.back().polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
          {0, 1, 2}, {Real{0}, Real{-3}, Real{-2}, Real{0}, Real{3}, Real{-2},
                      Real{0}, Real{0}, Real{3}});
  result.emplace_back();
  result.back().polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
          {0, 1, 2}, {Real{-2}, Real{0}, Real{-3}, Real{3}, Real{0}, Real{-1},
                      Real{-1}, Real{0}, Real{3}});
  return result;
}

template <typename Real>
auto combined_three_way_triangles() -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8},
      {Real{-3}, Real{-2}, Real{0},  Real{3},  Real{-2}, Real{0}, Real{0},
       Real{3},  Real{0},  Real{0},  Real{-3}, Real{-2}, Real{0}, Real{3},
       Real{-2}, Real{0},  Real{0},  Real{3},  Real{-2}, Real{0}, Real{-3},
       Real{3},  Real{0},  Real{-1}, Real{-1}, Real{0},  Real{3}});
  return result;
}

template <typename Real> auto octahedron() -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4},
      {Real{0}, Real{0}, Real{-1.3}, Real{1.4}, Real{0}, Real{0}, Real{0},
       Real{1.1}, Real{0}, Real{-0.9}, Real{0}, Real{0}, Real{0}, Real{-1.6},
       Real{0}, Real{0}, Real{0}, Real{1.8}});
  return result;
}

template <typename Real>
auto overlapping_boxes()
    -> std::pair<parity_operand<Real>, parity_operand<Real>> {
  parity_operand<Real> first;
  first.polygons = tf::cpp::make_box_mesh<tf::cpp::default_index_t>(
      Real{1}, Real{1}, Real{1});
  parity_operand<Real> second;
  second.polygons = tf::cpp::make_box_mesh<tf::cpp::default_index_t>(
      Real{1}, Real{0.75}, Real{0.5});
  second.place({Real{1}, Real{0}, Real{0}, Real{0.5}, Real{0}, Real{1}, Real{0},
                Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0},
                Real{0}, Real{1}});
  return {std::move(first), std::move(second)};
}

} // namespace tf::cpp::test
