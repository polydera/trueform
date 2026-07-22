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
#include "./construct/make_polygon_arrangement_index_map.hpp"
#include "./construct/make_polygon_arrangements.hpp"
#include "./make_intersection_curves.hpp"
#include "./make_arrangement_graph.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace cut {

/// Shared worker for the single-mesh arrangement entry points. `Curves` and
/// `IndexMap` are each either `tf::none_t` (skip) or their request tag; the
/// returned tuple grows accordingly. Keeps every public overload a one-line
/// dispatch and the pipeline body in one place.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P>
auto polygon_arrangements_worker(const P &p,
                                 tf::arrangement_config config) {
  // the factory tags what is missing; the graph is the type authority
  // and its form view carries the structures every consumer needs
  auto graph = tf::make_arrangement_graph<Int>(p, config);
  using Index = typename decltype(graph)::index_type;
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;
  auto form = graph.forms()[0];
  auto [mesh, face_labels, map_data] =
      tf::cut::make_polygon_arrangements<OutputCoordinateType>(
          form, graph.triangulations(), graph.converter(),
          graph.created_points());

  if constexpr (want_imap) {
    auto imap = tf::cut::make_polygon_arrangement_index_map(
        std::move(map_data), std::move(face_labels),
        static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap),
                             graph.converter());
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
    // curves are the non-manifold edges of the regions — the same
    // arrangement read every path uses
    auto cb = tf::make_intersection_curves<RealOut>(graph);
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             std::move(cb), graph.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             graph.converter());
    else
      return std::make_pair(std::move(mesh), std::move(face_labels));
  }
}

} // namespace cut

/// @ingroup cut_boolean
/// @brief Split a single mesh at its self-intersection curves.
///
/// Returns the mesh with faces split along self-intersection curves,
/// plus face labels mapping each output face to its original face index.
///
/// @tparam Policy The policy type of the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param config The intersection mode flags.
/// @return Tuple of (@ref tf::polygons_buffer, face labels).
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::arrangement_config config = {
                                   tf::intersect_mode::primitives |
                                   tf::intersect_mode::resolve_contours |
                                   tf::intersect_mode::within}) {
  return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::none_t, tf::none_t>(_polygons,
                                                                  config);
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::arrangement_config config,
                               tf::return_curves_t) {
  return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::return_curves_t, tf::none_t>(
      _polygons, config);
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::return_curves_t) {
  return make_polygon_arrangements<Int, OutputCoordinateType>(
      _polygons,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours |
                           tf::intersect_mode::within},
      tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves, returning a
///        @ref tf::polygon_arrangement_index_map relating the output back to
///        the input mesh.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::arrangement_config config,
                               tf::return_index_map_t) {
  return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::none_t, tf::return_index_map_t>(
      _polygons, config);
}

/// @ingroup cut_boolean
/// @brief Index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::return_index_map_t) {
  return make_polygon_arrangements<Int, OutputCoordinateType>(
      _polygons,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours |
                           tf::intersect_mode::within},
      tf::return_index_map);
}

} // namespace tf
