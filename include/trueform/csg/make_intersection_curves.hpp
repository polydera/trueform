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
#include "../cut/make_intersection_curves.hpp"
#include "./csg_graph.hpp"

namespace tf {

/// @ingroup csg
/// @brief The intersection-curve network of an N-form CSG arrangement.
///
/// An intersection edge is a region walk edge that has a neighbour of
/// a different form tag — i.e. where two surfaces cross. Endpoints are
/// created intersection vertices, whose `vertex_t.id` indexes straight
/// into `created_points()`. The edges are connected into polylines via
/// @ref tf::connect_edges_to_paths.
///
/// A coincident (coplanar) overlap contributes its contact border; the
/// overlap's interior stays silent (the duplicate walks are collapsed
/// out of the connectivity).
///
/// Forwards to the arrangement read with the collapsed connectivity
/// the classification already built.
///
/// @return A @ref tf::curves_buffer of the intersection polylines.
template <typename Forms, typename Structs, typename Int>
auto make_intersection_curves(const tf::csg_graph<Forms, Structs, Int> &graph) {
  using graph_t = tf::csg_graph<Forms, Structs, Int>;
  using RealOut = typename graph_t::input_real_type;

  return tf::cut::make_intersection_curves<RealOut>(
      graph.arrangement(),
      graph.labels().connectivity().connectivity_per_carrier_edge());
}

} // namespace tf
