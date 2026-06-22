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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/curves_buffer.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../intersect/intersect_config.hpp"
#include "../reindex/return_index_map.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/make_polygon_arrangement_index_map.hpp"
#include "./construct/make_polygon_arrangements.hpp"
#include "./dispatch/build_self_pipeline.hpp"
#include "./dispatch/self_boolean.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace cut {

/// Shared worker for the single-mesh arrangement entry points. `Curves` and
/// `IndexMap` are each either `tf::none_t` (skip) or their request tag; the
/// returned tuple grows accordingly. Keeps every public overload a one-line
/// dispatch and the pipeline body in one place.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P>
auto polygon_arrangements_worker(const P &p, tf::intersect_config config) {
  using Index = std::decay_t<decltype(p.faces()[0][0])>;
  using InputReal = tf::coordinate_type<std::decay_t<decltype(p)>>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;

  auto [iwp, ig, fc, cg] =
      cut::dispatch::build_self_pipeline<Index, PipelineReal, ResolvedInt>(
          p, config);
  auto [mesh, face_labels, map_data] =
      tf::cut::make_polygon_arrangements<OutputCoordinateType>(
          p, ig, fc, iwp.converter());

  if constexpr (want_imap) {
    auto imap = tf::cut::make_polygon_arrangement_index_map(
        std::move(map_data), std::move(face_labels),
        static_cast<Index>(mesh.points_buffer().size()));
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap), iwp.converter());
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (want_curves) {
    auto paths =
        tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
    auto &conv = iwp.converter();
    auto ipts = ig.points();
    tf::curves_buffer<Index, RealOut, 3> cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(ipts.size());
    if constexpr (std::is_integral_v<RealOut>) {
      tf::parallel_copy(tf::make_points(ipts), cb.points());
    } else {
      tf::parallel_copy(
          tf::make_points(tf::make_mapped_range(
              ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
          cb.points());
    }
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             std::move(cb), iwp.converter());
    else
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             std::move(cb));
  } else {
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(face_labels),
                             iwp.converter());
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
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_polygon_arrangements(
    const tf::polygons<Policy> &_polygons,
    tf::intersect_config config = {tf::intersect_mode::primitives |
                                   tf::intersect_mode::resolve_contours}) {
  return cut::dispatch::self_boolean(_polygons, [config](const auto &p) {
    return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                            tf::none_t, tf::none_t>(p, config);
  });
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::intersect_config config,
                               tf::return_curves_t) {
  return cut::dispatch::self_boolean(_polygons, [config](const auto &p) {
    return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                            tf::return_curves_t, tf::none_t>(
        p, config);
  });
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::return_curves_t) {
  return make_polygon_arrangements<Int, OutputCoordinateType>(
      _polygons,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours},
      tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Split a single mesh at self-intersection curves, returning a
///        @ref tf::polygon_arrangement_index_map relating the output back to
///        the input mesh.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::intersect_config config,
                               tf::return_index_map_t) {
  return cut::dispatch::self_boolean(_polygons, [config](const auto &p) {
    return cut::polygon_arrangements_worker<Int, OutputCoordinateType,
                                            tf::none_t, tf::return_index_map_t>(
        p, config);
  });
}

/// @ingroup cut_boolean
/// @brief Index-map overload with the default intersect mode.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_polygon_arrangements(const tf::polygons<Policy> &_polygons,
                               tf::return_index_map_t) {
  return make_polygon_arrangements<Int, OutputCoordinateType>(
      _polygons,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours},
      tf::return_index_map);
}

} // namespace tf
