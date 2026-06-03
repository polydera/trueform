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
#include "./csg_graph.hpp"
#include "./expression/compile.hpp"
#include "./expression/compiled_expr.hpp"
#include "./expression/expr.hpp"
#include "./graph/compute_chosen_sides.hpp"
#include "./graph/evaluate_per_domain.hpp"
#include "./graph/make_csg_mesh.hpp"
#include <type_traits>

namespace tf {

/// @ingroup csg
/// @brief Build the CSG result mesh for a boolean `e` evaluated
///        against `graph`.
///
/// One-stop user entry: compiles the expression, evaluates per
/// domain, picks per-component sides, and triangulates the boundary.
///
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, which resolves to the graph's
///         @c input_real_type (the input forms' coordinate type).
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_mesh(const tf::csg_graph<Forms, Structs, Int> &graph,
                   const tf::csg::expr &e) {
  using InputReal =
      typename tf::csg_graph<Forms, Structs, Int>::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;

  auto E = e.compile().evaluator();
  auto membership = tf::csg::graph::evaluate_per_domain(graph.inclusion(), E);
  auto chosen =
      tf::csg::graph::compute_chosen_sides(graph.descriptor(), membership);
  return tf::csg::graph::make_csg_mesh<RealOut>(graph.arrangement(),
                                           graph.face_cuts(),
                                           graph.intersection_graph(),
                                           graph.forms(), chosen,
                                           graph.converter());
}

} // namespace tf
