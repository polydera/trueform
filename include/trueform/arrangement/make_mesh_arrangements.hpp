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
#include "../core/resolved_output_real.hpp"
#include "../reindex/return_index_map.hpp"
#include "./arrangement_config.hpp"
#include "./make_arrangement_graph.hpp"
#include "./make_arrangement_mesh.hpp"
#include "./make_intersection_curves.hpp"
#include "./return_curves.hpp"

namespace tf {

namespace arrangement {

/// Shared worker for every arrangement entry point. `Curves` /
/// `IndexMap` are each `tf::none_t` (skip) or their request tag; the returned
/// tuple grows accordingly. Takes the built graph, so the operand shape —
/// one form, two typed forms, or a homogeneous range — is already settled by
/// whichever `make_arrangement_graph` overload the caller reached, and the
/// result shape follows from it: a single form reports no tag axis.
template <typename OutputCoordinateType, typename Curves, typename IndexMap,
          typename Graph>
auto arrangement_worker(const Graph &graph) {
  using InputReal = typename Graph::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  constexpr bool want_curves = !std::is_same_v<Curves, tf::none_t>;
  constexpr bool want_imap = !std::is_same_v<IndexMap, tf::none_t>;
  constexpr bool with_conv =
      !std::is_integral_v<InputReal> && std::is_integral_v<RealOut>;
  auto &conv = graph.converter();

  if constexpr (want_imap) {
    auto [mesh, imap] = tf::make_arrangement_mesh<OutputCoordinateType>(
        graph, tf::return_index_map);
    if constexpr (with_conv)
      return std::make_tuple(std::move(mesh), std::move(imap), conv);
    else
      return std::make_pair(std::move(mesh), std::move(imap));
  } else if constexpr (Graph::static_n_tags == 1) {
    // One operand has no tag axis, so the source-ids read is a pair.
    auto [mesh, face_labels] = tf::make_arrangement_mesh<OutputCoordinateType>(
        graph, tf::return_source_ids);
    if constexpr (want_curves) {
      auto cb = tf::make_intersection_curves<RealOut>(graph);
      if constexpr (with_conv)
        return std::make_tuple(std::move(mesh), std::move(face_labels),
                               std::move(cb), conv);
      else
        return std::make_tuple(std::move(mesh), std::move(face_labels),
                               std::move(cb));
    } else {
      if constexpr (with_conv)
        return std::make_tuple(std::move(mesh), std::move(face_labels), conv);
      else
        return std::make_pair(std::move(mesh), std::move(face_labels));
    }
  } else {
    auto [mesh, tag_labels, face_labels] =
        tf::make_arrangement_mesh<OutputCoordinateType>(graph,
                                                        tf::return_source_ids);
    if constexpr (want_curves) {
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
}

/// Two typed forms.
template <typename Int, typename OutputCoordinateType, typename Curves,
          typename IndexMap, typename P0, typename P1>
auto mesh_arrangements_pair_worker(const P0 &p0, const P1 &p1,
                                   tf::arrangement_config config) {
  return arrangement_worker<OutputCoordinateType, Curves, IndexMap>(
      tf::make_arrangement_graph<Int>(p0, p1, config));
}

/// A homogeneous range of forms.
template <typename Int, typename OutputCoordinateType,
          typename Curves = tf::none_t, typename IndexMap = tf::none_t,
          typename FormsRange>
auto mesh_arrangements_n(const FormsRange &forms,
                         tf::arrangement_config config) {
  return arrangement_worker<OutputCoordinateType, Curves, IndexMap>(
      tf::make_arrangement_graph<Int>(forms, config));
}

} // namespace arrangement

/// @ingroup arrangement_mesh
/// @brief Decompose two meshes into classified regions.
///
/// With two meshes the crossing flags are no-ops, so the pair defaults to
/// bare primitives.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::arrangement_config config =
                                tf::intersect_config{}) {
  return arrangement::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::none_t, tf::none_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup arrangement_mesh
/// @brief Decompose two meshes with curve output.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::arrangement_config config,
                            tf::return_curves_t) {
  return arrangement::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::return_curves_t, tf::none_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup arrangement_mesh
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

/// @ingroup arrangement_mesh
/// @brief Decompose two meshes, returning a @ref tf::mesh_arrangement_index_map
///        relating the output back to the input meshes.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1,
                            tf::arrangement_config config,
                            tf::return_index_map_t) {
  return arrangement::mesh_arrangements_pair_worker<Int, OutputCoordinateType,
                                            tf::none_t,
                                            tf::return_index_map_t>(
      _polygons0, _polygons1, config);
}

/// @ingroup arrangement_mesh
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

/// @ingroup arrangement_mesh
/// @brief Build a single merged mesh from N intersected meshes.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(
    const Range &_forms, tf::arrangement_config config = {}) {
  return arrangement::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                  tf::none_t>(_forms, config);
}

/// @ingroup arrangement_mesh
/// @brief Build a merged mesh with intersection curves.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &_forms,
                            tf::arrangement_config config,
                            tf::return_curves_t) {
  return arrangement::mesh_arrangements_n<Int, OutputCoordinateType,
                                  tf::return_curves_t, tf::none_t>(_forms,
                                                                   config);
}

/// @ingroup arrangement_mesh
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

/// @ingroup arrangement_mesh
/// @brief Build a merged mesh from N meshes, returning a
///        @ref tf::mesh_arrangement_index_map relating output back to input.
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Range>
auto make_mesh_arrangements(const Range &_forms,
                            tf::arrangement_config config,
                            tf::return_index_map_t) {
  return arrangement::mesh_arrangements_n<Int, OutputCoordinateType, tf::none_t,
                                  tf::return_index_map_t>(_forms, config);
}

/// @ingroup arrangement_mesh
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
