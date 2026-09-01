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
#include "../reindex/return_index_map.hpp"
#include "./arrangement_config.hpp"
#include "./make_arrangement_graph.hpp"
#include "./make_intersection_curves.hpp"
#include "./make_mesh_arrangements.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace arrangement {

/// Shared worker for the single-mesh arrangement entry points. `Curves` and
/// `IndexMap` are each either `tf::none_t` (skip) or their request tag; the
/// returned tuple grows accordingly. Keeps every public overload a one-line
/// dispatch and the pipeline body in one place.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P>
auto polygon_arrangements_worker(const P &p, tf::arrangement_config config) {
  return arrangement_worker<OutputCoordinateType, Curves, IndexMap>(
      tf::make_arrangement_graph<Int>(p, config));
}

} // namespace arrangement

/// @ingroup arrangement_mesh
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
  return arrangement::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::none_t, tf::none_t>(_polygons,
                                                                  config);
}

/// @ingroup arrangement_mesh
/// @brief Split a single mesh at self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::arrangement_config config,
                               tf::return_curves_t) {
  return arrangement::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::return_curves_t, tf::none_t>(
      _polygons, config);
}

/// @ingroup arrangement_mesh
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

/// @ingroup arrangement_mesh
/// @brief Split a single mesh at self-intersection curves, returning a
///        @ref tf::polygon_arrangement_index_map relating the output back to
///        the input mesh.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::arrangement_config config,
                               tf::return_index_map_t) {
  return arrangement::polygon_arrangements_worker<Int, OutputCoordinateType,
                                          tf::none_t, tf::return_index_map_t>(
      _polygons, config);
}

/// @ingroup arrangement_mesh
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
