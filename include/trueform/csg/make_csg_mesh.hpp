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
#include "../arrangement/construct/make_mesh_arrangement_index_map.hpp"
#include "../arrangement/make_arrangement_mesh.hpp"
#include "../core/none.hpp"
#include "../core/resolved_output_real.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/return_source_ids.hpp"
#include "./csg_graph.hpp"
#include "./expression/selection.hpp"
#include "./graph/chosen_sides_for.hpp"
#include "./graph/make_csg_mesh.hpp"
#include <tuple>
#include <type_traits>

namespace tf {

/// @ingroup csg
/// @brief Build the CSG result mesh for a selection evaluated against
///        `graph` — a boolean expression, a surface read, or a boolean
///        restricted to the surfaces that contributed.
///
/// One-stop user entry. A plain @ref tf::csg::expr converts implicitly
/// (every form's surface, the boolean's classification); the
/// @ref tf::csg::selection factory names the forms — a braced list or
/// range alone is the embedded read (their surfaces cut by everything,
/// original winding), and paired with an expression it restricts the
/// boolean's result to the faces those forms contributed, so one result
/// splits by provenance.
///
/// @ref tf::csg::inside names the same forms and expression as the other
/// read of them: the surfaces lying INSIDE the expression's region —
/// both sides of a piece in it — emitted with the form's stored winding.
/// Every overload below accepts either kind.
///
/// @pre The graph holds two or more forms. Expressions combine forms;
///      a one-form graph is a self arrangement — read it with
///      @ref tf::make_outer_shell / @ref tf::make_csg_domains (the
///      expression-less arrangement overloads below remain valid).
///
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, which resolves to the graph's
///         @c input_real_type (the input forms' coordinate type).
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                   const tf::csg::selection_t &s) {
  using InputReal =
      typename tf::csg_graph<Policy, Int, Arrangement>::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto chosen = tf::csg::graph::chosen_sides_for(graph, s);
  return tf::csg::graph::make_csg_mesh<
      RealOut, tf::csg_graph<Policy, Int, Arrangement>::face_static_size>(
      graph.arrangement(), graph.labels(), chosen,
      s.mask(graph.arrangement().n_tags()));
}

/// @ingroup csg
/// @brief Build the CSG result mesh for `e`, additionally returning per
///        output face its provenance: `tag_labels[f]` = which input form
///        face `f` came from, `face_labels[f]` = the original face id within
///        that form. Mirrors the (tag_labels, face_labels) pair returned by
///        @ref tf::make_mesh_arrangements.
///
/// @return Tuple of (@ref tf::polygons_buffer, tag_labels, face_labels).
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                   const tf::csg::selection_t &s, tf::return_source_ids_t) {
  using InputReal =
      typename tf::csg_graph<Policy, Int, Arrangement>::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto chosen = tf::csg::graph::chosen_sides_for(graph, s);
  auto [mesh, tag_labels, face_labels, uncut_faces, map_data] =
      tf::csg::graph::make_csg_mesh<
          RealOut, tf::csg_graph<Policy, Int, Arrangement>::face_static_size,
          /*WantLabels=*/true>(graph.arrangement(), graph.labels(), chosen,
                               s.mask(graph.arrangement().n_tags()));
  (void)map_data;
  (void)uncut_faces;
  return std::make_tuple(std::move(mesh), std::move(tag_labels),
                         std::move(face_labels));
}

/// @ingroup csg
/// @brief Boolean result mesh for `e` with a @ref tf::mesh_arrangement_index_map
///        relating every output point and face back to the input forms.
///
/// Same map contract as the full-arrangement overload: output -> (input form,
/// input element) inverse for points and faces, the input -> output forward
/// point map, and the `end` sentinel for created intersection points and for
/// input points that no surviving face kept.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                   const tf::csg::selection_t &s, tf::return_index_map_t) {
  using Index = typename tf::csg_graph<Policy, Int, Arrangement>::index_type;
  using InputReal =
      typename tf::csg_graph<Policy, Int, Arrangement>::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto chosen = tf::csg::graph::chosen_sides_for(graph, s);
  auto [mesh, tag_labels, face_labels, uncut_faces, map_data] =
      tf::csg::graph::make_csg_mesh<
          RealOut, tf::csg_graph<Policy, Int, Arrangement>::face_static_size,
          /*WantLabels=*/true>(graph.arrangement(), graph.labels(), chosen,
                               s.mask(graph.arrangement().n_tags()));
  const Index n_out = static_cast<Index>(mesh.points_buffer().size());
  auto imap = tf::arrangement::make_mesh_arrangement_index_map(
      std::move(map_data), std::move(tag_labels), std::move(face_labels),
      std::move(uncut_faces), n_out);
  return std::make_pair(std::move(mesh), std::move(imap));
}

/// @ingroup csg
/// @brief Build the full arrangement mesh of `graph` — every input face,
///        cut at intersections, each surface emitted once — without a
///        boolean selection.
///
/// This is the graph analogue of @ref tf::make_mesh_arrangements: it reuses
/// the intersection graph and face cuts the graph already holds, so only the
/// mesh is materialised — the intersection pipeline is not re-run. Mirrors
/// how @ref tf::make_csg_domains with no expression returns every domain.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph) {
  return tf::make_arrangement_mesh<OutputCoordinateType>(graph.arrangement());
}

/// @ingroup csg
/// @brief Full arrangement mesh with per-face provenance: `tag_labels[f]` =
///        which input form, `face_labels[f]` = original face id within it.
///
/// @return Tuple of (@ref tf::polygons_buffer, tag_labels, face_labels).
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                   tf::return_source_ids_t) {
  return tf::make_arrangement_mesh<OutputCoordinateType>(
      graph.arrangement(), tf::return_source_ids);
}

/// @ingroup csg
/// @brief Full arrangement mesh with a @ref tf::mesh_arrangement_index_map
///        relating every output point and face back to the input forms.
///
/// The map folds in the `(tag, face)` provenance and adds the point axis:
/// output point -> (input form, input point) inverse, and the input ->
/// output forward map (`point_f`). Created intersection points carry the
/// `end` sentinel (no input origin). Same contract as
/// @ref tf::make_mesh_arrangements with @ref tf::return_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_mesh(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                   tf::return_index_map_t) {
  return tf::make_arrangement_mesh<OutputCoordinateType>(
      graph.arrangement(), tf::return_index_map);
}

} // namespace tf
