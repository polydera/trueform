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
#include "./construct/make_polygon_arrangement_index_map.hpp"
#include "./construct/make_uncut_face_ranges.hpp"
#include <tuple>
#include <utility>

namespace tf {

/// @ingroup arrangement
/// @brief Materialise the full arrangement mesh of `graph` — every
///        input face, cut at intersections, each surface emitted once.
///
/// Reuses the triangle stream the graph already holds, so only the mesh
/// is materialised — the arrangement pipeline is not re-run.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(const tf::arrangement_graph<Policy, Int> &graph) {
  auto result =
      tf::arrangement::make_mesh_arrangements<OutputCoordinateType>(graph);
  return std::move(std::get<0>(result)); // mesh; drop labels + map_data
}

/// @ingroup arrangement
/// @brief Arrangement mesh with per-face provenance: `tag_labels[f]` =
///        which input form, `face_labels[f]` = original face id within
///        it.
///
/// A one-form arrangement has no tag axis to report — every face comes
/// from the one operand — so it returns only the face ids.
///
/// @return Tuple of (@ref tf::polygons_buffer, tag_labels, face_labels),
///         or (@ref tf::polygons_buffer, face_labels) for one form.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(const tf::arrangement_graph<Policy, Int> &graph,
                           tf::return_source_ids_t) {
  auto result =
      tf::arrangement::make_mesh_arrangements<OutputCoordinateType>(graph);
  if constexpr (tf::arrangement_graph<Policy, Int>::static_n_tags == 1)
    return std::make_pair(std::move(std::get<0>(result)),
                          std::move(std::get<2>(result)));
  else
    return std::make_tuple(std::move(std::get<0>(result)),
                           std::move(std::get<1>(result)),
                           std::move(std::get<2>(result)));
}

/// @ingroup arrangement
/// @brief Arrangement mesh with a @ref tf::mesh_arrangement_index_map
///        relating every output point and face back to the input forms.
///
/// The map folds in the `(tag, face)` provenance and adds the point
/// axis: output point -> (input form, input point) inverse, and the
/// input -> output forward map (`point_f`). Created intersection
/// points carry the `end` sentinel (no input origin).
///
/// Which map that is follows the operand count, which the graph's
/// policy knows at compile time: one form has no tag axis, so it gets a
/// @ref tf::polygon_arrangement_index_map; two or more get the
/// @ref tf::mesh_arrangement_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_arrangement_mesh(const tf::arrangement_graph<Policy, Int> &graph,
                           tf::return_index_map_t) {
  using Index = typename tf::arrangement_graph<Policy, Int>::index_type;
  auto [mesh, tag_labels, face_labels, map_data] =
      tf::arrangement::make_mesh_arrangements<OutputCoordinateType>(graph);
  const Index n_original_faces = map_data.total_original_faces;
  const Index n_out = static_cast<Index>(mesh.points_buffer().size());
  if constexpr (tf::arrangement_graph<Policy, Int>::static_n_tags == 1) {
    (void)tag_labels;
    auto imap = tf::arrangement::make_polygon_arrangement_index_map(
        std::move(map_data), std::move(face_labels), n_original_faces, n_out);
    return std::make_pair(std::move(mesh), std::move(imap));
  } else {
    auto uncut_faces = tf::arrangement::make_uncut_face_ranges(map_data);
    auto imap = tf::arrangement::make_mesh_arrangement_index_map(
        std::move(map_data), std::move(tag_labels), std::move(face_labels),
        std::move(uncut_faces), n_out);
    return std::make_pair(std::move(mesh), std::move(imap));
  }
}

} // namespace tf
