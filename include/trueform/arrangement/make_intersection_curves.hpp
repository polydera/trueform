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
#include "../core/range.hpp"
#include "../core/resolved_output_real.hpp"
#include "./arrangement_graph.hpp"
#include "./construct/extract_intersection_curves.hpp"

namespace tf {

/// @ingroup arrangement
/// @brief The intersection-curve network of an arrangement.
///
/// A seam is a constraint edge of the exposed triangle stream whose two
/// endpoints are created vertices and whose incidences do not all carry
/// one tag; within a single tag (a self arrangement) it is a
/// non-manifold edge. The contact border of a coincident (coplanar)
/// overlap is a seam in both cases, while the overlap's interior stays
/// silent — the stack mask decides it.
///
/// @tparam OutputCoordinateType Output coordinate type (defaulted to the
///         input's).
/// @return A @ref tf::curves_buffer of the connected polylines.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int>
auto make_intersection_curves(const tf::arrangement_graph<Policy, Int> &graph) {
  using graph_t = tf::arrangement_graph<Policy, Int>;
  using Index = typename graph_t::index_type;
  using InputReal = typename graph_t::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;

  const auto &ga = graph.global();
  return tf::arrangement::extract_triangle_seam_curves<RealOut, Index>(
      ga.exposed_tris(), ga.exposed_cons_bits(), graph.triangle_tags(),
      graph.dead(), graph.stacked(), graph.n_tags(),
      tf::make_range(graph.created_points()), graph.converter());
}

} // namespace tf
