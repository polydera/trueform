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
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/resolved_output_real.hpp"
#include "../intersect/intersect_config.hpp"
#include "../intersect/intersect_mode.hpp"
#include "./csg_graph.hpp"
#include "./expression/selection_kind.hpp"
#include "./graph/compute_chosen_sides.hpp"
#include "./graph/make_csg_mesh.hpp"
#include "./graph/structural_membership.hpp"
#include "./make_csg_graph.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup csg
/// @brief Outer shell of a csg graph: the boundary of the union of
/// everything its forms enclose.
///
/// Structural read — no winding bits, no boolean expression. The
/// unbounded universe is the most-negative-volume domain (the seeder's
/// oracle, stored on the graph); the shell is the boundary between the
/// universe and everything else. This is the immersion-safe read: a
/// self-intersecting form's double-covered pockets stay enclosed, where
/// a winding-parity bit would drop them. The natural extraction for a
/// one-form graph (the self arrangement), valid for any N.
///
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, which resolves to the graph's input type.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_outer_shell(const tf::csg_graph<Policy, Int, Arrangement> &graph) {
  using InputReal =
      typename tf::csg_graph<Policy, Int, Arrangement>::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto membership = tf::csg::graph::compute_structural_membership(
      graph.domain_volumes(), graph.domain_nesting_merges());

  auto chosen = tf::csg::graph::compute_chosen_sides(
      graph.descriptor(), membership, tf::csg::selection_kind::boundary);
  return tf::csg::graph::make_csg_mesh<
      RealOut, tf::csg_graph<Policy, Int, Arrangement>::face_static_size>(
      graph.arrangement(), graph.labels(), chosen);
}

/// @ingroup csg
/// @brief Repair a mesh to its outer shell: the boundary of the union of
/// everything it encloses.
///
/// The mesh's own @ref tf::csg_graph — its self arrangement plus the
/// classification tier — answered by the structural read above. Internal
/// structure (overlap membranes between interpenetrating parts, faces
/// buried inside the solid, enclosed cavities) has the same domain on
/// both sides and so never reaches the boundary; open fragments are
/// self-merged by the arrangement and are non-separating by
/// construction. The result is free of self-intersections and suitable
/// as a boolean or csg-graph operand.
///
/// @tparam Int Exact integer type for the arrangement (deduced by default).
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, which resolves to the input mesh's real type.
///         An integral type keeps the shell on the exact lattice the
///         arrangement ran on, and the converter that produced it is
///         returned alongside.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param config The intersection mode flags for the self-arrangement.
/// @return A @ref tf::polygons_buffer bounding the enclosed union, or a
///         tuple of it and the converter for integer output.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_outer_shell(const tf::polygons<Policy> &_polygons,
                      tf::intersect_config config = {
                          tf::intersect_mode::primitives |
                          tf::intersect_mode::resolve_contours}) {
  using InputReal = tf::coordinate_type<Policy>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  auto graph = tf::make_csg_graph<Int>(_polygons, config);
  auto shell = tf::make_outer_shell<RealOut>(graph);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(shell), graph.converter());
  } else {
    return shell;
  }
}

} // namespace tf
