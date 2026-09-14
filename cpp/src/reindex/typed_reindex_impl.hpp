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

#include "trueform/cpp/reindex/by_ids.hpp"
#include "trueform/cpp/reindex/by_ids_on_points.hpp"
#include "trueform/cpp/reindex/by_mask.hpp"
#include "trueform/cpp/reindex/by_mask_on_points.hpp"
#include "trueform/cpp/reindex/concatenated.hpp"
#include "trueform/cpp/reindex/fixed_arity.hpp"
#include "trueform/cpp/reindex/mesh.hpp"
#include "trueform/cpp/reindex/selection_result.hpp"
#include "trueform/cpp/reindex/split_into_domains.hpp"

#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/core/views/slice.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/reindex/detail/array_mesh.hpp"
#include "trueform/reindex.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace detail {

template <typename Index, typename Real, std::size_t Dims>
auto require_reindex_axes() -> void {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "reindex Real must be float or double");
  static_assert(is_supported_index_v<Index>,
                "reindex Index must be int32 or int64");
  static_assert(Dims == 2 || Dims == 3, "reindex Dims must be two or three");
}

template <std::size_t Dims, typename Real>
auto require_points(const nd_array<Real> &points, const char *operation)
    -> void {
  if (!points.is_valid())
    throw std::invalid_argument(std::string(operation) +
                                ": points must be valid");
  if (points.ndim() != 2 || points.shape_at(1) != static_cast<int>(Dims))
    throw std::invalid_argument(std::string(operation) +
                                ": points must have shape [N, Dims]");
}

template <typename Selection>
auto require_selection_vector(const Selection &selection, std::size_t expected,
                              const char *operation, const char *name) -> void {
  if (!selection.is_valid())
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " must be valid");
  if (selection.ndim() != 1)
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " must be one-dimensional");
  if (selection.length() != expected)
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " size mismatch");
}

inline auto require_mask(const nd_array<std::int8_t> &mask,
                         std::size_t expected, const char *operation) -> void {
  require_selection_vector(mask, expected, operation, "mask");
  const auto refused = tf::parallel_contains(
      mask.make_range(),
      [](std::int8_t value) {
        return value != std::int8_t{0} && value != std::int8_t{1};
      },
      tf::checked);
  if (refused)
    throw std::invalid_argument(std::string(operation) +
                                ": mask values must be zero or one");
}

template <typename Index>
auto require_ids(const nd_array<Index> &ids, std::size_t count,
                 const char *operation) -> void {
  if (!ids.is_valid() || ids.ndim() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": ids must be a valid one-dimensional array");
  if (count > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    throw std::length_error(std::string(operation) +
                            ": source exceeds index range");
  tf::buffer<bool> seen;
  seen.allocate(count);
  tf::parallel_fill(seen, false);
  for (const auto id : ids) {
    if (id < Index{0} || static_cast<std::size_t>(id) >= count)
      throw std::out_of_range(std::string(operation) + ": id out of range");
    const auto offset = static_cast<std::size_t>(id);
    if (seen[offset])
      throw std::invalid_argument(std::string(operation) + ": duplicate id");
    seen[offset] = true;
  }
}

template <typename Index>
auto point_ids_to_mask(const nd_array<Index> &ids, std::size_t count,
                       const char *operation) -> nd_array<std::int8_t> {
  if (!ids.is_valid() || ids.ndim() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": ids must be a valid one-dimensional array");
  if (count > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
      count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(std::string(operation) +
                            ": source exceeds index range");

  tf::buffer<std::int8_t> values;
  values.allocate(count);
  tf::parallel_fill(values, std::int8_t{0});
  for (const auto id : ids) {
    if (id < Index{0} || static_cast<std::size_t>(id) >= count)
      throw std::out_of_range(std::string(operation) + ": id out of range");
    values[static_cast<std::size_t>(id)] = std::int8_t{1};
  }
  return nd_array<std::int8_t>::from_buffer(std::move(values),
                                            {static_cast<int>(count)});
}

template <typename Index>
auto require_index_map(const index_map<Index> &map, std::size_t source_count,
                       const char *operation, const char *name) -> void {
  if (!map.is_valid() || map.f.ndim() != 1 || map.kept_ids.ndim() != 1)
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " must contain one-dimensional arrays");
  if (map.f.length() != source_count)
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " forward-map size mismatch");
  if (map.kept_ids.length() > source_count)
    throw std::invalid_argument(std::string(operation) + ": " + name +
                                " kept-id size mismatch");
  if (source_count >
      static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    throw std::length_error(std::string(operation) + ": " + name +
                            " exceeds index range");

  const auto sentinel = static_cast<Index>(source_count);
  tf::buffer<bool> covered;
  covered.allocate(map.kept_ids.length());
  tf::parallel_fill(covered, false);
  for (const auto mapped : map.f) {
    if (mapped == sentinel)
      continue;
    if (mapped < Index{0} ||
        static_cast<std::size_t>(mapped) >= map.kept_ids.length())
      throw std::out_of_range(std::string(operation) + ": " + name +
                              " forward value out of range");
    covered[static_cast<std::size_t>(mapped)] = true;
  }
  for (const auto present : covered)
    if (!present)
      throw std::invalid_argument(std::string(operation) + ": " + name +
                                  " does not cover every output index");
  for (std::size_t mapped = 0; mapped < map.kept_ids.length(); ++mapped) {
    const auto source = map.kept_ids[mapped];
    if (source < Index{0} || static_cast<std::size_t>(source) >= source_count)
      throw std::out_of_range(std::string(operation) + ": " + name +
                              " kept id out of range");
    if (map.f[static_cast<std::size_t>(source)] != static_cast<Index>(mapped))
      throw std::invalid_argument(std::string(operation) + ": " + name +
                                  " arrays are inconsistent");
  }
}

template <typename Index>
auto make_index_map_view(const index_map<Index> &map) {
  return tf::make_index_map(map.f.make_range(), map.kept_ids.make_range());
}

/// The reindex axes are a compile-time fact about this module, so an edge
/// entry states them beside the reading it takes; the indices are the cache's
/// fact, asked of it.
template <typename Index, typename Real, std::size_t Dims>
auto require_reindexed_edge_mesh(const edge_mesh<Index, Real, Dims> &value)
    -> void {
  require_reindex_axes<Index, Real, Dims>();
  value.require_indices();
}

template <typename Real, std::size_t Dims>
auto points_array(tf::points_buffer<Real, Dims> &&points) -> nd_array<Real> {
  const auto count = static_cast<int>(points.size());
  return nd_array<Real>::from_buffer(std::move(points.data_buffer()),
                                     {count, static_cast<int>(Dims)});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Tuple>
auto make_typed_mesh_result(Tuple &&result)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  auto [polygons, face_map, point_map] = std::forward<Tuple>(result);
  return {std::move(polygons),
          index_map<Index>::from_index_map_buffer(std::move(face_map)),
          index_map<Index>::from_index_map_buffer(std::move(point_map))};
}

template <typename Index, typename Real, std::size_t Dims, typename Tuple>
auto make_typed_edge_result(Tuple &&result)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  auto [segments, edge_map, point_map] = std::forward<Tuple>(result);
  return {std::move(segments),
          index_map<Index>::from_index_map_buffer(std::move(edge_map)),
          index_map<Index>::from_index_map_buffer(std::move(point_map))};
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Kernel>
auto select_mesh_ids(const mesh<Index, Real, Dims, Ngon> &value,
                     const nd_array<Index> &ids, const char *operation,
                     bool points, Kernel kernel)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  value.require_indices();
  require_ids(ids, points ? value.number_of_points() : value.number_of_faces(),
              operation);
  return make_typed_mesh_result<Index, Real, Dims, Ngon>(
      kernel(value.polygons(), ids.make_range()));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Kernel>
auto select_mesh_mask(const mesh<Index, Real, Dims, Ngon> &value,
                      const nd_array<std::int8_t> &mask, const char *operation,
                      bool points, Kernel kernel)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  value.require_indices();
  require_mask(mask,
               points ? value.number_of_points() : value.number_of_faces(),
               operation);
  return make_typed_mesh_result<Index, Real, Dims, Ngon>(
      kernel(value.polygons(), mask.make_range()));
}

template <typename Index, typename Real, std::size_t Dims, typename Kernel>
auto select_edge_ids(const edge_mesh<Index, Real, Dims> &value,
                     const nd_array<Index> &ids, const char *operation,
                     bool points, Kernel kernel)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  require_reindexed_edge_mesh(value);
  require_ids(ids,
              points ? value.number_of_points() : value.number_of_edges(),
              operation);
  return make_typed_edge_result<Index, Real, Dims>(
      kernel(value.segments(), ids.make_range()));
}

template <typename Index, typename Real, std::size_t Dims, typename Kernel>
auto select_edge_mask(const edge_mesh<Index, Real, Dims> &value,
                      const nd_array<std::int8_t> &mask, const char *operation,
                      bool points, Kernel kernel)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  require_reindexed_edge_mesh(value);
  require_mask(mask,
               points ? value.number_of_points() : value.number_of_edges(),
               operation);
  return make_typed_edge_result<Index, Real, Dims>(
      kernel(value.segments(), mask.make_range()));
}

template <typename Index>
auto checked_add(std::size_t total, std::size_t add, const char *operation)
    -> std::size_t {
  constexpr auto index_max =
      static_cast<std::size_t>(std::numeric_limits<Index>::max());
  constexpr auto public_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (add > index_max - total || add > public_max - total)
    throw std::length_error(std::string(operation) +
                            ": output exceeds index range");
  return total + add;
}

/// A view's frame is always there — identity where the owner carries none —
/// so a concatenation places every operand the same way and no branch decides
/// which of two copies runs.
template <typename View, typename Points, typename Destination>
auto copy_points(const View &view, const Points &points,
                 Destination destination) -> void {
  tf::reindex::copy_with_transformation(points, destination, view.frame());
}

} // namespace detail

// Explicit map application ---------------------------------------------------

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed(const mesh<Index, Real, Dims, Ngon> &value,
               const index_map<Index> &face_map,
               const index_map<Index> &point_map)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  const auto face_view = detail::make_index_map_view(face_map);
  const auto point_view = detail::make_index_map_view(point_map);
  value.require_indices();
  detail::require_index_map(face_map, value.number_of_faces(), "reindexed",
                            "face map");
  detail::require_index_map(point_map, value.number_of_points(), "reindexed",
                            "point map");
  const auto point_sentinel = static_cast<Index>(value.number_of_points());
  const auto faces = value.faces();
  for (const auto face_id : face_map.kept_ids)
    for (const auto point_id : faces[face_id])
      if (point_map.f[static_cast<std::size_t>(point_id)] == point_sentinel)
        throw std::invalid_argument(
            "reindexed: selected face references a removed point");
  return tf::reindexed(value.polygons(), face_view, point_view);
}

// Raw points -----------------------------------------------------------------

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_with_maps(const nd_array<Real> &points,
                                const nd_array<Index> &ids)
    -> reindexed_points_result<Index, Real> {
  detail::require_reindex_axes<Index, Real, Dims>();
  detail::require_points<Dims>(points, "reindexed_by_ids");
  detail::require_ids(ids, static_cast<std::size_t>(points.shape_at(0)),
                      "reindexed_by_ids");
  auto [output, map] =
      tf::reindexed_by_ids<Index>(tf::make_points<Dims>(points.make_range()),
                                  ids.make_range(), tf::return_index_map);
  return {detail::points_array<Real, Dims>(std::move(output)),
          index_map<Index>::from_index_map_buffer(std::move(map))};
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids(const nd_array<Real> &points, const nd_array<Index> &ids)
    -> nd_array<Real> {
  return reindexed_by_ids_with_maps<Index, Real, Dims>(points, ids).points;
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask_with_maps(const nd_array<Real> &points,
                                 const nd_array<std::int8_t> &mask)
    -> reindexed_points_result<Index, Real> {
  detail::require_reindex_axes<Index, Real, Dims>();
  detail::require_points<Dims>(points, "reindexed_by_mask");
  detail::require_mask(mask, static_cast<std::size_t>(points.shape_at(0)),
                       "reindexed_by_mask");
  auto [output, map] =
      tf::reindexed_by_mask<Index>(tf::make_points<Dims>(points.make_range()),
                                   mask.make_range(), tf::return_index_map);
  return {detail::points_array<Real, Dims>(std::move(output)),
          index_map<Index>::from_index_map_buffer(std::move(map))};
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask(const nd_array<Real> &points,
                       const nd_array<std::int8_t> &mask) -> nd_array<Real> {
  return reindexed_by_mask_with_maps<Index, Real, Dims>(points, mask).points;
}

// Mesh selectors -------------------------------------------------------------

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids_with_maps(const mesh<Index, Real, Dims, Ngon> &value,
                                const nd_array<Index> &ids)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  return detail::select_mesh_ids(
      value, ids, "reindexed_by_ids", false,
      [](const auto &polygons, const auto &selection) {
        return tf::reindexed_by_ids<Index>(polygons, selection,
                                           tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids(const mesh<Index, Real, Dims, Ngon> &value,
                      const nd_array<Index> &ids)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return cpp::reindexed_by_ids_with_maps(value, ids).mesh;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_mask_with_maps(const mesh<Index, Real, Dims, Ngon> &value,
                                 const nd_array<std::int8_t> &mask)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  return detail::select_mesh_mask(
      value, mask, "reindexed_by_mask", false,
      [](const auto &polygons, const auto &selection) {
        return tf::reindexed_by_mask<Index>(polygons, selection,
                                            tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_mask(const mesh<Index, Real, Dims, Ngon> &value,
                       const nd_array<std::int8_t> &mask)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return cpp::reindexed_by_mask_with_maps(value, mask).mesh;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids_on_points_with_maps(
    const mesh<Index, Real, Dims, Ngon> &value, const nd_array<Index> &ids)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  const auto mask = detail::point_ids_to_mask(
      ids, static_cast<std::size_t>(value.number_of_points()),
      "reindexed_by_ids_on_points");
  return cpp::reindexed_by_mask_on_points_with_maps(value, mask);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids_on_points(const mesh<Index, Real, Dims, Ngon> &value,
                                const nd_array<Index> &ids)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return cpp::reindexed_by_ids_on_points_with_maps(value, ids).mesh;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_mask_on_points_with_maps(
    const mesh<Index, Real, Dims, Ngon> &value,
    const nd_array<std::int8_t> &mask)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon> {
  return detail::select_mesh_mask(
      value, mask, "reindexed_by_mask_on_points", true,
      [](const auto &polygons, const auto &selection) {
        return tf::reindexed_by_mask_on_points<Index>(polygons, selection,
                                                      tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_mask_on_points(const mesh<Index, Real, Dims, Ngon> &value,
                                 const nd_array<std::int8_t> &mask)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  return cpp::reindexed_by_mask_on_points_with_maps(value, mask).mesh;
}

// Edge-mesh selectors --------------------------------------------------------

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_with_maps(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<Index> &ids)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  return detail::select_edge_ids(
      value, ids, "reindexed_by_ids", false,
      [](const auto &segments, const auto &selection) {
        return tf::reindexed_by_ids<Index>(segments, selection,
                                           tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids(const edge_mesh<Index, Real, Dims> &value,
                      const nd_array<Index> &ids)
    -> tf::segments_buffer<Index, Real, Dims> {
  return reindexed_by_ids_with_maps<Index, Real, Dims>(value, ids).mesh;
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask_with_maps(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<std::int8_t> &mask)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  return detail::select_edge_mask(
      value, mask, "reindexed_by_mask", false,
      [](const auto &segments, const auto &selection) {
        return tf::reindexed_by_mask<Index>(segments, selection,
                                            tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask(const edge_mesh<Index, Real, Dims> &value,
                       const nd_array<std::int8_t> &mask)
    -> tf::segments_buffer<Index, Real, Dims> {
  return reindexed_by_mask_with_maps<Index, Real, Dims>(value, mask).mesh;
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points_with_maps(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<Index> &ids)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  detail::require_reindexed_edge_mesh(value);
  const auto mask = detail::point_ids_to_mask(ids, value.number_of_points(),
                                              "reindexed_by_ids_on_points");
  return reindexed_by_mask_on_points_with_maps<Index, Real, Dims>(value, mask);
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<Index> &ids) -> tf::segments_buffer<Index, Real, Dims> {
  return reindexed_by_ids_on_points_with_maps<Index, Real, Dims>(value, ids)
      .mesh;
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask_on_points_with_maps(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<std::int8_t> &mask)
    -> reindexed_edge_mesh_result<Index, Real, Dims> {
  return detail::select_edge_mask(
      value, mask, "reindexed_by_mask_on_points", true,
      [](const auto &segments, const auto &selection) {
        return tf::reindexed_by_mask_on_points<Index>(segments, selection,
                                                      tf::return_index_map);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_mask_on_points(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<std::int8_t> &mask)
    -> tf::segments_buffer<Index, Real, Dims> {
  return reindexed_by_mask_on_points_with_maps<Index, Real, Dims>(value, mask)
      .mesh;
}

// Tuple-equivalent array inputs ---------------------------------------------

namespace detail {

/// A tuple-equivalent entry states the arity its arrays carry, so the carrier
/// its selection runs on is a compile-time choice between the two.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename Selection, typename Kernel>
auto select_fixed_arity(const nd_array<Index> &elements,
                        const nd_array<Real> &points,
                        const Selection &selection, Kernel kernel) {
  static_assert(Vertices == 2 || Vertices == 3,
                "fixed tuple arity must be two or three");
  if constexpr (Vertices == 2) {
    array_edge_mesh<Index, Real, Dims> carrier(elements, points);
    return kernel(carrier.edge_mesh(), selection);
  } else {
    array_mesh<Index, Real, Dims> carrier(elements, points);
    return kernel(carrier.mesh(), selection);
  }
}

} // namespace detail

#define TF_CPP_REINDEX_FIXED_ARITY_SELECTION(Name, SelectionType, ResultType)  \
  template <typename Index, typename Real, std::size_t Dims,                   \
            std::size_t Vertices,                                              \
            std::enable_if_t<Vertices == 2 || Vertices == 3, int>>             \
  auto Name(const nd_array<Index> &elements, const nd_array<Real> &points,     \
            const nd_array<SelectionType> &selection)                          \
      ->ResultType<Index, Real, Dims, Vertices> {                              \
    return detail::select_fixed_arity<Index, Real, Dims, Vertices>(            \
        elements, points, selection,                                           \
        [](const auto &carrier, const auto &chosen) {                          \
          return cpp::Name(carrier, chosen);                                   \
        });                                                                    \
  }

TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_ids, Index,
                                     reindexed_fixed_carrier_t)
TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_ids_with_maps, Index,
                                     reindexed_fixed_result_t)
TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_mask, std::int8_t,
                                     reindexed_fixed_carrier_t)
TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_mask_with_maps, std::int8_t,
                                     reindexed_fixed_result_t)
TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_ids_on_points, Index,
                                     reindexed_fixed_carrier_t)
TF_CPP_REINDEX_FIXED_ARITY_SELECTION(reindexed_by_mask_on_points, std::int8_t,
                                     reindexed_fixed_carrier_t)

#undef TF_CPP_REINDEX_FIXED_ARITY_SELECTION

/// A mesh stated as arrays is one reading of them, so the entry assembles it
/// over the caller's own storage and answers through the compiled carrier
/// entry; the cache it needed lives for exactly this call. A carrier spelled
/// with two template arguments cannot be a macro argument, so each layout is
/// its own macro.
#define TF_CPP_REINDEX_TRIANGLE_ARRAY_SELECTION(Name, SelectionType,           \
                                                ResultType)                    \
  template <typename Index, typename Real, std::size_t Dims>                   \
  auto Name(const nd_array<Index> &faces, const nd_array<Real> &points,        \
            const nd_array<SelectionType> &selection)                          \
      ->ResultType<Index, Real, Dims, 3> {                                     \
    detail::array_mesh<Index, Real, Dims, 3> carrier(faces, points);           \
    return cpp::Name(carrier.mesh(), selection);                               \
  }

#define TF_CPP_REINDEX_MIXED_ARRAY_SELECTION(Name, SelectionType, ResultType)  \
  template <typename Index, typename Real, std::size_t Dims>                   \
  auto Name(const offset_blocked_buffer<Index, Index> &faces,                  \
            const nd_array<Real> &points,                                      \
            const nd_array<SelectionType> &selection)                          \
      ->ResultType<Index, Real, Dims, tf::dynamic_size> {                      \
    detail::array_mesh<Index, Real, Dims, tf::dynamic_size> carrier(faces,     \
                                                                    points);   \
    return cpp::Name(carrier.mesh(), selection);                               \
  }

#define TF_CPP_REINDEX_ARRAY_SELECTIONS(Selection)                             \
  Selection(reindexed_by_ids, Index, tf::polygons_buffer)                      \
  Selection(reindexed_by_ids_with_maps, Index, reindexed_mesh_result)          \
  Selection(reindexed_by_mask, std::int8_t, tf::polygons_buffer)               \
  Selection(reindexed_by_mask_with_maps, std::int8_t, reindexed_mesh_result)   \
  Selection(reindexed_by_ids_on_points, Index, tf::polygons_buffer)            \
  Selection(reindexed_by_mask_on_points, std::int8_t, tf::polygons_buffer)

TF_CPP_REINDEX_ARRAY_SELECTIONS(TF_CPP_REINDEX_TRIANGLE_ARRAY_SELECTION)
TF_CPP_REINDEX_ARRAY_SELECTIONS(TF_CPP_REINDEX_MIXED_ARRAY_SELECTION)

#undef TF_CPP_REINDEX_ARRAY_SELECTIONS
#undef TF_CPP_REINDEX_MIXED_ARRAY_SELECTION
#undef TF_CPP_REINDEX_TRIANGLE_ARRAY_SELECTION


// Concatenation --------------------------------------------------------------

/// A range is homogeneous, so its element's arity is the result's, and the two
/// layouts are two loops rather than a promotion.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto concatenate_meshes(
    const std::vector<mesh<Index, Real, Dims, Ngon>> &operands)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  detail::require_reindex_axes<Index, Real, Dims>();
  if (operands.empty())
    throw std::invalid_argument("concatenate_meshes: meshes must not be empty");

  std::size_t total_faces = 0;
  std::size_t total_indices = 0;
  std::size_t total_points = 0;
  for (const auto &operand : operands) {
    operand.require_indices();
    total_faces = detail::checked_add<Index>(total_faces,
                                             operand.number_of_faces(),
                                             "concatenate_meshes");
    total_indices = detail::checked_add<Index>(
        total_indices, operand.geometry().indices.size(),
        "concatenate_meshes");
    total_points = detail::checked_add<Index>(total_points,
                                              operand.number_of_points(),
                                              "concatenate_meshes");
  }

  tf::polygons_buffer<Index, Real, Dims, Ngon> output;
  output.points_buffer().allocate(total_points);
  std::size_t point_at = 0;
  if constexpr (Ngon == 3) {
    output.faces_buffer().allocate(total_faces);
    std::size_t face_at = 0;
    for (const auto &operand : operands) {
      for (const auto face : operand.faces()) {
        auto destination = output.faces_buffer()[face_at++];
        for (std::size_t corner = 0; corner < 3; ++corner)
          destination[corner] =
              static_cast<Index>(face[corner] + static_cast<Index>(point_at));
      }
      const auto end = point_at + operand.points().size();
      detail::copy_points(operand, operand.points(),
                          tf::slice(output.points_buffer(), point_at, end));
      point_at = end;
    }
  } else {
    auto &offsets = output.faces_buffer().offsets_buffer();
    auto &indices = output.faces_buffer().data_buffer();
    offsets.allocate(total_faces + 1);
    indices.allocate(total_indices);
    offsets[0] = Index{0};
    std::size_t face_at = 0;
    std::size_t index_at = 0;
    for (const auto &operand : operands) {
      for (const auto face : operand.faces()) {
        for (const auto point_id : face)
          indices[index_at++] =
              static_cast<Index>(point_id + static_cast<Index>(point_at));
        offsets[++face_at] = static_cast<Index>(index_at);
      }
      const auto end = point_at + operand.points().size();
      detail::copy_points(operand, operand.points(),
                          tf::slice(output.points_buffer(), point_at, end));
      point_at = end;
    }
  }
  return output;
}

template <typename Index, typename Real, std::size_t Dims>
auto concatenate_edge_meshes(
    const std::vector<edge_mesh<Index, Real, Dims>> &operands)
    -> tf::segments_buffer<Index, Real, Dims> {
  detail::require_reindex_axes<Index, Real, Dims>();
  if (operands.empty())
    throw std::invalid_argument(
        "concatenate_edge_meshes: meshes must not be empty");

  std::size_t total_edges = 0;
  std::size_t total_points = 0;
  for (const auto &operand : operands) {
    detail::require_reindexed_edge_mesh(operand);
    total_edges = detail::checked_add<Index>(total_edges,
                                             operand.number_of_edges(),
                                             "concatenate_edge_meshes");
    total_points = detail::checked_add<Index>(total_points,
                                              operand.number_of_points(),
                                              "concatenate_edge_meshes");
  }

  tf::segments_buffer<Index, Real, Dims> output;
  output.edges_buffer().allocate(total_edges);
  output.points_buffer().allocate(total_points);
  std::size_t edge_at = 0;
  std::size_t point_at = 0;
  for (const auto &operand : operands) {
    for (const auto edge : operand.edges()) {
      auto destination = output.edges_buffer()[edge_at++];
      destination[0] =
          static_cast<Index>(edge[0] + static_cast<Index>(point_at));
      destination[1] =
          static_cast<Index>(edge[1] + static_cast<Index>(point_at));
    }
    const auto points = operand.points();
    const auto end = point_at + points.size();
    detail::copy_points(operand, points,
                        tf::slice(output.points_buffer(), point_at, end));
    point_at = end;
  }
  return output;
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto concatenate_meshes(const mesh<Index0, Real0, Dims, Ngon0> &operand0,
                        const mesh<Index1, Real1, Dims, Ngon1> &operand1)
    -> tf::polygons_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims,
                           detail::concatenated_arity_v<Ngon0, Ngon1>> {
  using Index = common_index_t<Index0, Index1>;
  detail::require_reindex_axes<Index0, Real0, Dims>();
  detail::require_reindex_axes<Index1, Real1, Dims>();

  operand0.require_indices();
  operand1.require_indices();
  auto total_faces = detail::checked_add<Index>(0, operand0.number_of_faces(),
                                                "concatenate_meshes");
  detail::checked_add<Index>(total_faces, operand1.number_of_faces(),
                             "concatenate_meshes");
  auto total_points = detail::checked_add<Index>(0, operand0.number_of_points(),
                                                 "concatenate_meshes");
  detail::checked_add<Index>(total_points, operand1.number_of_points(),
                             "concatenate_meshes");
  return tf::concatenated(operand0.polygons() | tf::tag(operand0.frame()),
                          operand1.polygons() | tf::tag(operand1.frame()));
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto concatenate_edge_meshes(const edge_mesh<Index0, Real0, Dims> &operand0,
                             const edge_mesh<Index1, Real1, Dims> &operand1)
    -> tf::segments_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims> {
  using Index = common_index_t<Index0, Index1>;
  detail::require_reindexed_edge_mesh(operand0);
  detail::require_reindexed_edge_mesh(operand1);

  auto total_edges = detail::checked_add<Index>(0, operand0.number_of_edges(),
                                                "concatenate_edge_meshes");
  detail::checked_add<Index>(total_edges, operand1.number_of_edges(),
                             "concatenate_edge_meshes");
  auto total_points = detail::checked_add<Index>(0, operand0.number_of_points(),
                                                 "concatenate_edge_meshes");
  detail::checked_add<Index>(total_points, operand1.number_of_points(),
                             "concatenate_edge_meshes");

  return tf::concatenated(operand0.segments() | tf::tag(operand0.frame()),
                          operand1.segments() | tf::tag(operand1.frame()));
}

// Domain splitting -----------------------------------------------------------

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_domains(const mesh<Index, Real, Dims, Ngon> &value,
                        const domain_labels_result<Index> &labels)
    -> split_domains_result<Index, Real, Dims, Ngon> {
  detail::require_reindex_axes<Index, Real, Dims>();
  if (labels.number_of_faces() != static_cast<Index>(value.number_of_faces()))
    throw std::invalid_argument(
        "split_into_domains: labels must have one row per face");

  const auto domain_count = labels.number_of_domains();
  if (domain_count < Index{0})
    throw std::invalid_argument(
        "split_into_domains: domain count must be nonnegative");
  const auto label_values = labels.labels();
  if (!label_values.is_valid() || label_values.ndim() != 2 ||
      label_values.shape_at(1) != 2)
    throw std::invalid_argument(
        "split_into_domains: labels must have shape [N, 2]");
  if (static_cast<std::uint64_t>(domain_count) >
      static_cast<std::uint64_t>(label_values.length()))
    throw std::invalid_argument(
        "split_into_domains: domain count exceeds label entries");
  tf::buffer<bool> seen;
  seen.allocate(static_cast<std::size_t>(domain_count));
  tf::parallel_fill(seen, false);
  for (const auto label : label_values) {
    if (label < Index{0} || label > domain_count)
      throw std::out_of_range("split_into_domains: label out of range");
    if (label < domain_count)
      seen[static_cast<std::size_t>(label)] = true;
  }
  for (const auto present : seen)
    if (!present)
      throw std::invalid_argument(
          "split_into_domains: every domain must have a labeled side");

  tf::domain_labels<Index> native_labels;
  native_labels.n_domains = domain_count;
  native_labels.outer_shell_label = labels.outer_shell_label();
  native_labels.labels.allocate(value.number_of_faces());
  tf::parallel_copy(label_values.make_range(),
                    native_labels.labels.data_buffer());

  value.require_indices();
  auto [components, component_labels] =
      tf::split_into_domains(value.polygons(), native_labels);
  std::vector<tf::polygons_buffer<Index, Real, Dims, Ngon>> result_components;
  result_components.reserve(components.size());
  for (auto &component : components)
    result_components.push_back(std::move(component));
  const auto label_count = static_cast<int>(component_labels.size());
  return {std::move(result_components),
          nd_array<Index>::from_buffer(std::move(component_labels),
                                       {label_count})};
}

} // namespace tf::cpp
