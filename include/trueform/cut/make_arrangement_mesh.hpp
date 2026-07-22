/*
 * Copyright (c) 2026 XLAB
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
#include "../core/none.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/return_source_ids.hpp"
#include "./arrangement_graph.hpp"
#include "./construct/make_mesh_arrangement_index_map.hpp"
#include "./construct/make_mesh_arrangements.hpp"
#include <tuple>
#include <utility>

namespace tf {

/// @ingroup cut
/// @brief Materialise the full arrangement mesh of `graph` — every
///        input face, cut at intersections, each surface emitted once.
///
/// Reuses the intersection graph and face cuts the graph already
/// holds, so only the mesh is materialised — the intersection pipeline
/// is not re-run.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(const tf::arrangement_graph<Policy, Int> &graph) {
  auto result = tf::cut::make_mesh_arrangements<OutputCoordinateType>(
      graph.triangulations(), graph.forms(), graph.converter(),
      graph.created_points());
  return std::move(std::get<0>(result)); // mesh; drop labels + map_data
}

/// @ingroup cut
/// @brief Arrangement mesh with per-face provenance: `tag_labels[f]` =
///        which input form, `face_labels[f]` = original face id within
///        it.
///
/// @return Tuple of (@ref tf::polygons_buffer, tag_labels, face_labels).
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(const tf::arrangement_graph<Policy, Int> &graph,
                           tf::return_source_ids_t) {
  auto result = tf::cut::make_mesh_arrangements<OutputCoordinateType>(
      graph.triangulations(), graph.forms(), graph.converter(),
      graph.created_points());
  return std::make_tuple(std::move(std::get<0>(result)),
                         std::move(std::get<1>(result)),
                         std::move(std::get<2>(result)));
}

/// @ingroup cut
/// @brief Arrangement mesh with a @ref tf::mesh_arrangement_index_map
///        relating every output point and face back to the input forms.
///
/// The map folds in the `(tag, face)` provenance and adds the point
/// axis: output point -> (input form, input point) inverse, and the
/// input -> output forward map (`point_f`). Created intersection
/// points carry the `end` sentinel (no input origin).
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(
    const tf::arrangement_graph<Policy, Int> &graph,
    tf::return_index_map_t) {
  using Index = typename tf::arrangement_graph<Policy, Int>::index_type;
  auto [mesh, tag_labels, face_labels, map_data] =
      tf::cut::make_mesh_arrangements<OutputCoordinateType>(
          graph.triangulations(), graph.forms(), graph.converter(),
          graph.created_points());
  const Index n_original_faces = map_data.total_original_faces;
  const Index n_out = static_cast<Index>(mesh.points_buffer().size());
  auto imap = tf::cut::make_mesh_arrangement_index_map(
      std::move(map_data), std::move(tag_labels), std::move(face_labels),
      n_original_faces, n_out);
  return std::make_pair(std::move(mesh), std::move(imap));
}

} // namespace tf
