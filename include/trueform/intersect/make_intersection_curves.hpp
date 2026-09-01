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
#include "../arrangement/arrangement_config.hpp"
#include "../arrangement/make_arrangement_graph.hpp"
#include "../arrangement/make_intersection_curves.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/resolved_output_real.hpp"
#include "./intersect_config.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

namespace intersect {

/// Shared worker for the standalone curve entry points: the arrangement
/// already carries the seam read, so a curves-only caller differs from an
/// arrangement caller in nothing but what it keeps. An integer-coordinate
/// output also hands back the converter that produced the lattice.
template <typename OutputCoordinateType, typename Graph>
auto curves_worker(const Graph &graph) {
  using InputReal = typename Graph::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto cb = tf::make_intersection_curves<RealOut>(graph);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(cb), graph.converter());
  } else {
    return cb;
  }
}

} // namespace intersect

/// @ingroup intersect_curves
/// @brief Extract intersection curves where two meshes intersect.
///
/// Computes the geometric intersection between two polygon meshes and
/// returns the result as connected curves.
///
/// @tparam Policy0 The policy type for the first mesh.
/// @tparam Policy1 The policy type for the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param config The intersection configuration (mode and tolerance).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_intersection_curves(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_config config = {tf::intersect_mode::primitives}) {
  return intersect::curves_worker<OutputCoordinateType>(
      tf::make_arrangement_graph<Int>(_polygons0, _polygons1,
                                       tf::arrangement_config{config}));
}

/// @ingroup intersect_curves
/// @brief Extract intersection curves from N meshes.
///
/// Computes all pairwise intersections among the given meshes and
/// returns the result as connected curves.
///
/// @tparam Range A range of @ref tf::polygons (or tagged forms).
/// @param _forms The input meshes.
/// @param config The intersection configuration (mode and tolerance).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_intersection_curves(
    const Range &_forms,
    tf::intersect_config config = {tf::intersect_mode::primitives |
                                   tf::intersect_mode::resolve_crossing_contours}) {
  return intersect::curves_worker<OutputCoordinateType>(
      tf::make_arrangement_graph<Int>(_forms, tf::arrangement_config{config}));
}

} // namespace tf
