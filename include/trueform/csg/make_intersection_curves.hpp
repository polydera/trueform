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
#include "../arrangement/make_intersection_curves.hpp"
#include "../core/none.hpp"
#include "../core/resolved_output_real.hpp"
#include "./csg_graph.hpp"
#include <type_traits>

namespace tf {

/// @ingroup csg
/// @brief The intersection-curve network of an N-form CSG arrangement.
///
/// An intersection edge is a constraint edge of the exposed triangle
/// stream whose incidences do not all carry one tag — i.e. where two
/// surfaces cross. Endpoints are created intersection vertices, whose
/// `vertex_t.id` indexes straight into `created_points()`. The edges are
/// connected into polylines via @ref tf::connect_edges_to_paths.
///
/// A coincident (coplanar) overlap contributes its contact border; the
/// overlap's interior stays silent.
///
/// @return A @ref tf::curves_buffer of the intersection polylines.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_intersection_curves(
    const tf::csg_graph<Policy, Int, Arrangement> &graph) {
  using graph_t = tf::csg_graph<Policy, Int, Arrangement>;
  using InputReal = typename graph_t::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  return tf::make_intersection_curves<RealOut>(graph.arrangement());
}

} // namespace tf
