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
#include "./arrangement_config.hpp"
#include "../reindex/return_index_map.hpp"
#include "./make_intersection_curves.hpp"
#include "./construct/make_mesh_arrangement_index_map.hpp"
#include "./construct/make_mesh_arrangements.hpp"
#include "./make_arrangement_graph.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace cut {

/// Shared worker for the two-mesh arrangement entry points. `Curves` /
/// `IndexMap` are each `tf::none_t` (skip) or their request tag; the returned
/// tuple grows accordingly. Body lives once; public overloads dispatch here.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P0, typename P1>
auto mesh_arrangements_pair_worker(const P0 &p0, const P1 &p1,
                                   tf::arrangement_config config) {
  // the factory tags what is missing; the policy owns it. The graph is
  // the type authority, and extraction below reads geometry only —
  // every structure-dependent read goes through the graph.
  auto graph = tf::make_arrangement_graph<Int>(p0, p1, config);
  using Index = typename decltype(graph)::index_type;
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;
  auto &conv = graph.converter();

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<OutputCoordinateType>(
          graph.triangulations(), p0, p1, conv, graph.created_points());

  if constexpr (want_imap) {
    const auto n_original_faces = map_data.total_original_faces;
    auto imap = tf::cut::make_mesh_arrangement_index_map(
        std::move(map_data), std::move(tag_labels), std::move(face_labels),
        n_original_faces, static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap), conv);
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
    auto cb = tf::make_intersection_curves<RealOut>(graph);
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb), conv);
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), conv);
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels));
  }
}

/// Shared worker for the N-mesh arrangement entry point. `Curves`/`IndexMap`
/// default to `tf::none_t` (plain), so existing 2-arg-template callers keep
/// working. Folds the curve build (SoS or cut-graph) and the index map into
/// one body.
template <typename Int, typename OutputCoordinateType,
          typename Curves = tf::none_t, typename IndexMap = tf::none_t,
          typename FormsRange>
auto mesh_arrangements_n(const FormsRange &forms,
                         tf::arrangement_config config) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;

  auto graph = tf::make_arrangement_graph<Int>(forms, config);
  auto &conv = graph.converter();

  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<RealOut>(graph.triangulations(), forms,
                                               conv, graph.created_points());

  if constexpr (want_imap) {
    const auto n_original_faces = map_data.total_original_faces;
    auto imap = make_mesh_arrangement_index_map(
        std::move(map_data), std::move(tag_labels), std::move(face_labels),
        n_original_faces, static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap),
                             graph.converter());
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
    auto cb = tf::make_intersection_curves<RealOut>(graph);
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb),
                             graph.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), graph.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels));
  }
}

} // namespace cut

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            // pair keeps its historical bare-primitives
                            // default; with two meshes the crossing
                            // flags are no-ops anyway
                            tf::arrangement_config config =
                                tf::intersect_config{}) {
  return cut::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::none_t, tf::none_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curve output.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::arrangement_config config,
                            tf::return_curves_t) {
  return cut::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::return_curves_t, tf::none_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes with curves (default intersect mode).
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_config{}, tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Decompose two meshes, returning a @ref tf::mesh_arrangement_index_map
///        relating the output back to the input meshes.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::arrangement_config config,
                            tf::return_index_map_t) {
  return cut::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::none_t,
                                            tf::return_index_map_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup cut_boolean
/// @brief Two-mesh index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::return_index_map_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      _polygons0, _polygons1, tf::intersect_config{}, tf::return_index_map);
}

/// @ingroup cut_boolean
/// @brief Build a single merged mesh from N intersected meshes.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(
    const Range &_forms, tf::arrangement_config config = {}) {
  return cut::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                  tf::none_t>(_forms, config);
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with intersection curves.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &_forms,
                            tf::arrangement_config config,
                            tf::return_curves_t) {
  return cut::mesh_arrangements_n<Int, OutputCoordinateType,
                                  tf::return_curves_t, tf::none_t>(_forms,
                                                                   config);
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh with curves (default intersect mode).
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_curves_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      forms,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours},
      tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Build a merged mesh from N meshes, returning a
///        @ref tf::mesh_arrangement_index_map relating output back to input.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &_forms,
                            tf::arrangement_config config,
                            tf::return_index_map_t) {
  return cut::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                  tf::return_index_map_t>(_forms, config);
}

/// @ingroup cut_boolean
/// @brief N-mesh index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &forms, tf::return_index_map_t) {
  return make_mesh_arrangements<Int, OutputCoordinateType>(
      forms,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_crossing_contours},
      tf::return_index_map);
}

} // namespace tf
