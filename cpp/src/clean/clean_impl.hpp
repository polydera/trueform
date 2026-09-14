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

#include "trueform/cpp/clean/edge_mesh.hpp"
#include "trueform/cpp/clean/mesh.hpp"
#include "trueform/cpp/clean/points.hpp"
#include "trueform/cpp/clean/soup.hpp"

#include "trueform/clean/config.hpp"
#include "trueform/clean/points.hpp"
#include "trueform/clean/polygons.hpp"
#include "trueform/clean/segments.hpp"
#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/points_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

template <typename Real>
auto normalize_clean_tolerance(Real tolerance) -> Real {
  if (!std::isfinite(tolerance))
    throw std::invalid_argument("clean: tolerance must be finite");
  if (tolerance < Real{0})
    throw std::invalid_argument("clean: tolerance must be non-negative");
  return tolerance > Real{0} ? tolerance : Real{0};
}

template <std::size_t Dims, typename Real>
auto require_clean_points(const nd_array<Real> &points, const char *operation)
    -> void {
  static_assert(Dims == 2 || Dims == 3, "cleaned_points Dims must be 2 or 3");
  if (!points.is_valid())
    throw std::invalid_argument(std::string(operation) +
                                ": points must be valid");
  const auto single =
      points.ndim() == 1 && points.shape_at(0) == static_cast<int>(Dims);
  const auto batch =
      points.ndim() == 2 && points.shape_at(1) == static_cast<int>(Dims);
  if (!single && !batch)
    throw std::invalid_argument(
        std::string(operation) + ": points must have shape [" +
        std::to_string(Dims) + "] or [N, " + std::to_string(Dims) + "]");
}

template <std::size_t Dims, std::size_t Vertices, typename Real>
auto require_polygon_soup(const nd_array<Real> &polygons, const char *operation)
    -> void {
  static_assert(Dims == 2 || Dims == 3,
                "cleaned_polygon_soup Dims must be 2 or 3");
  static_assert(Vertices == 2 || Vertices == 3,
                "cleaned_polygon_soup Vertices must be 2 or 3");
  if (!polygons.is_valid())
    throw std::invalid_argument(std::string(operation) +
                                ": input must be valid");
  const auto single = polygons.ndim() == 2 &&
                      polygons.shape_at(0) == static_cast<int>(Vertices) &&
                      polygons.shape_at(1) == static_cast<int>(Dims);
  const auto batch = polygons.ndim() == 3 &&
                     polygons.shape_at(1) == static_cast<int>(Vertices) &&
                     polygons.shape_at(2) == static_cast<int>(Dims);
  if (!single && !batch)
    throw std::invalid_argument(
        std::string(operation) + ": input must have shape [" +
        std::to_string(Vertices) + ", " + std::to_string(Dims) + "] or [N, " +
        std::to_string(Vertices) + ", " + std::to_string(Dims) + "]");
}

template <std::size_t Dims, typename Real>
auto clean_points_array(tf::points_buffer<Real, Dims> &&points)
    -> nd_array<Real> {
  const auto count = static_cast<int>(points.size());
  return nd_array<Real>::from_buffer(std::move(points.data_buffer()),
                                     {count, static_cast<int>(Dims)});
}

/// The points a carrier stands on, deduplicated: the one kernel behind every
/// entry that cleans coordinates, whether they came as an array or as the
/// points of a cloud.
template <std::size_t Dims, typename Real, typename Points>
auto cleaned_points_of(const Points &input, Real tolerance) -> nd_array<Real> {
  if (tolerance > Real{0})
    return clean_points_array<Dims>(tf::cleaned(input, tolerance));
  return clean_points_array<Dims>(tf::cleaned(input));
}

template <std::size_t Dims, typename Real, typename Points>
auto cleaned_points_with_map_of(const Points &input, Real tolerance)
    -> cleaned_points_result<default_index_t, Real> {
  if (tolerance > Real{0}) {
    auto [result, point_map] =
        tf::cleaned(input, tolerance, tf::return_index_map);
    return {clean_points_array<Dims>(std::move(result)),
            index_map<>::from_index_map_buffer(std::move(point_map))};
  }
  auto [result, point_map] = tf::cleaned(input, tf::return_index_map);
  return {clean_points_array<Dims>(std::move(result)),
          index_map<>::from_index_map_buffer(std::move(point_map))};
}

template <typename Real, std::size_t Dims>
auto cleaned_points_array_impl(const nd_array<Real> &points, Real tolerance,
                               const char *operation) -> nd_array<Real> {
  require_clean_points<Dims>(points, operation);
  return cleaned_points_of<Dims, Real>(
      tf::make_points<Dims>(points.make_range()), tolerance);
}

template <typename Real, std::size_t Dims>
auto cleaned_points_with_map_impl(const nd_array<Real> &points, Real tolerance,
                                  const char *operation)
    -> cleaned_points_result<default_index_t, Real> {
  require_clean_points<Dims>(points, operation);
  return cleaned_points_with_map_of<Dims, Real>(
      tf::make_points<Dims>(points.make_range()), tolerance);
}

template <typename Index, typename Real, std::size_t Dims>
auto require_clean_axes() -> void {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "clean Real must be float or double");
  static_assert(is_supported_index_v<Index>,
                "clean Index must be int32 or int64");
  static_assert(Dims == 2 || Dims == 3, "clean Dims must be two or three");
}

template <typename Index> auto typed_empty_index_map() -> index_map<Index> {
  tf::index_map_buffer<Index> map;
  map.f().allocate(0);
  map.kept_ids().allocate(0);
  return index_map<Index>::from_index_map_buffer(std::move(map));
}

template <typename Index>
auto typed_removed_index_map(int source_size) -> index_map<Index> {
  tf::index_map_buffer<Index> map;
  map.f().allocate(static_cast<std::size_t>(source_size));
  map.kept_ids().allocate(0);
  tf::parallel_fill(map.f(), static_cast<Index>(source_size));
  return index_map<Index>::from_index_map_buffer(std::move(map));
}

template <typename Index, typename Real, std::size_t Dims>
struct cleaned_carrier_points {
  tf::points_buffer<Real, Dims> points;
  index_map<Index> point_map;
};

/// A carrier with no primitives is its points and nothing else: they all go
/// where the configuration drops what nothing references, and they deduplicate
/// where it does not.
template <typename Index, typename Real, std::size_t Dims, typename Points>
auto clean_carrier_points(const Points &points, Real tolerance,
                          bool remove_unreferenced_points)
    -> cleaned_carrier_points<Index, Real, Dims> {
  if (remove_unreferenced_points) {
    tf::points_buffer<Real, Dims> empty;
    empty.allocate(0);
    return {std::move(empty),
            typed_removed_index_map<Index>(static_cast<int>(points.size()))};
  }
  if (tolerance > Real{0}) {
    auto [result, point_map] =
        tf::cleaned<Index>(points, tolerance, tf::return_index_map);
    return {std::move(result),
            index_map<Index>::from_index_map_buffer(std::move(point_map))};
  }
  auto [result, point_map] = tf::cleaned<Index>(points, tf::return_index_map);
  return {std::move(result),
          index_map<Index>::from_index_map_buffer(std::move(point_map))};
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto clean_zero_face_mesh(const mesh<Index, Real, Dims, Ngon> &value,
                          Real tolerance, bool remove_unreferenced_points)
    -> cleaned_mesh_result<Index, Real, Dims, Ngon> {
  auto points = clean_carrier_points<Index, Real, Dims>(
      value.points(), tolerance, remove_unreferenced_points);
  tf::polygons_buffer<Index, Real, Dims, Ngon> result;
  result.points_buffer() = std::move(points.points);
  return {std::move(result), typed_empty_index_map<Index>(),
          std::move(points.point_map)};
}

template <typename Index, typename Real, std::size_t Dims>
auto clean_zero_edge_mesh(const edge_mesh<Index, Real, Dims> &value,
                          Real tolerance, bool remove_unreferenced_points)
    -> cleaned_edge_mesh_result<Index, Real, Dims> {
  auto points = clean_carrier_points<Index, Real, Dims>(
      value.points(), tolerance, remove_unreferenced_points);
  tf::segments_buffer<Index, Real, Dims> result;
  result.edges_buffer().allocate(0);
  result.points_buffer() = std::move(points.points);
  return {std::move(result), typed_empty_index_map<Index>(),
          std::move(points.point_map)};
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
auto cleaned_polygon_soup(const nd_array<Real> &polygons, Real tolerance, bool,
                          bool)
    -> basic_cleaned_polygon_soup_result<Index, Real, Dims, Vertices> {
  detail::require_clean_axes<Index, Real, Dims>();
  tolerance = detail::normalize_clean_tolerance(tolerance);
  detail::require_polygon_soup<Dims, Vertices>(polygons,
                                               "cleaned_polygon_soup");
  const auto points = tf::make_points<Dims>(polygons.make_range());
  if constexpr (Vertices == 2) {
    const auto soup =
        tf::make_segments(tf::make_blocked_range<Vertices>(points));
    if (tolerance > Real{0})
      return tf::cleaned<Index>(soup, tolerance);
    return tf::cleaned<Index>(soup);
  } else {
    static_assert(Vertices == 3,
                  "cleaned_polygon_soup Vertices must be 2 or 3");
    const auto soup =
        tf::make_polygons(tf::make_blocked_range<Vertices>(points));
    if (tolerance > Real{0})
      return tf::cleaned<Index>(soup, tolerance);
    return tf::cleaned<Index>(soup);
  }
}

template <typename Real, std::size_t Dims>
auto cleaned_points(const nd_array<Real> &points, Real tolerance, bool, bool)
    -> nd_array<Real> {
  tolerance = detail::normalize_clean_tolerance(tolerance);
  return detail::cleaned_points_array_impl<Real, Dims>(points, tolerance,
                                                       "cleaned_points");
}

template <typename Real, std::size_t Dims>
auto cleaned_points_with_map(const nd_array<Real> &points, Real tolerance, bool,
                             bool)
    -> cleaned_points_result<default_index_t, Real> {
  tolerance = detail::normalize_clean_tolerance(tolerance);
  return detail::cleaned_points_with_map_impl<Real, Dims>(
      points, tolerance, "cleaned_points_with_map");
}

template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int>>
auto cleaned_points(const point_cloud<Real, Dims> &value, Real tolerance, bool,
                    bool) -> nd_array<Real> {
  tolerance = detail::normalize_clean_tolerance(tolerance);
  return detail::cleaned_points_of<Dims, Real>(value.points(), tolerance);
}

template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int>>
auto cleaned_points_with_map(const point_cloud<Real, Dims> &value,
                             Real tolerance, bool, bool)
    -> cleaned_points_result<default_index_t, Real> {
  tolerance = detail::normalize_clean_tolerance(tolerance);
  return detail::cleaned_points_with_map_of<Dims, Real>(value.points(),
                                                        tolerance);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh(const mesh<Index, Real, Dims, Ngon> &value, Real tolerance,
                  bool remove_duplicate_primitives,
                  bool remove_unreferenced_points)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  detail::require_clean_axes<Index, Real, Dims>();
  tolerance = detail::normalize_clean_tolerance(tolerance);
  if (!value.number_of_faces())
    return detail::clean_zero_face_mesh(value, tolerance,
                                        remove_unreferenced_points)
        .mesh;

  const tf::clean_config_t<Real> config{tolerance, remove_duplicate_primitives,
                                        remove_unreferenced_points};
  value.require_indices();
  return tf::cleaned<Index>(value.polygons(), config);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh_with_maps(const mesh<Index, Real, Dims, Ngon> &value,
                            Real tolerance, bool remove_duplicate_primitives,
                            bool remove_unreferenced_points)
    -> cleaned_mesh_result<Index, Real, Dims, Ngon> {
  detail::require_clean_axes<Index, Real, Dims>();
  tolerance = detail::normalize_clean_tolerance(tolerance);
  if (!value.number_of_faces())
    return detail::clean_zero_face_mesh(value, tolerance,
                                        remove_unreferenced_points);

  const tf::clean_config_t<Real> config{tolerance, remove_duplicate_primitives,
                                        remove_unreferenced_points};
  value.require_indices();
  auto [result, face_map, point_map] =
      tf::cleaned<Index>(value.polygons(), config, tf::return_index_map);
  return {std::move(result),
          index_map<Index>::from_index_map_buffer(std::move(face_map)),
          index_map<Index>::from_index_map_buffer(std::move(point_map))};
}

template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh(const edge_mesh<Index, Real, Dims> &value,
                       Real tolerance, bool remove_duplicate_primitives,
                       bool remove_unreferenced_points)
    -> tf::segments_buffer<Index, Real, Dims> {
  detail::require_clean_axes<Index, Real, Dims>();
  tolerance = detail::normalize_clean_tolerance(tolerance);
  if (!value.number_of_edges())
    return detail::clean_zero_edge_mesh(value, tolerance,
                                        remove_unreferenced_points)
        .edge_mesh;

  const tf::clean_config_t<Real> config{tolerance, remove_duplicate_primitives,
                                        remove_unreferenced_points};
  value.require_indices();
  return tf::cleaned<Index>(value.segments(), config);
}

template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh_with_maps(const edge_mesh<Index, Real, Dims> &value,
                                 Real tolerance,
                                 bool remove_duplicate_primitives,
                                 bool remove_unreferenced_points)
    -> cleaned_edge_mesh_result<Index, Real, Dims> {
  detail::require_clean_axes<Index, Real, Dims>();
  tolerance = detail::normalize_clean_tolerance(tolerance);
  if (!value.number_of_edges())
    return detail::clean_zero_edge_mesh(value, tolerance,
                                        remove_unreferenced_points);

  const tf::clean_config_t<Real> config{tolerance, remove_duplicate_primitives,
                                        remove_unreferenced_points};
  value.require_indices();
  auto [result, edge_map, point_map] =
      tf::cleaned<Index>(value.segments(), config, tf::return_index_map);
  return {std::move(result),
          index_map<Index>::from_index_map_buffer(std::move(edge_map)),
          index_map<Index>::from_index_map_buffer(std::move(point_map))};
}

} // namespace tf::cpp
