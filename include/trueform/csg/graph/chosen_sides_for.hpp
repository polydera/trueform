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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../csg_graph.hpp"
#include "../expression/compile.hpp"
#include "../expression/compiled_expr.hpp"
#include "../expression/selection.hpp"
#include "./compute_chosen_sides.hpp"
#include "./evaluate_per_domain.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per-component sides for a selection: the expression's
///        classification under the selection's kind when it has one,
///        all-forward when it does not (the surface read keeps input
///        winding and consults nothing).
template <typename Policy, typename Int,
          template <typename, typename> class Arrangement>
auto chosen_sides_for(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                      const tf::csg::selection_t &s)
    -> tf::buffer<std::int8_t> {
  if (s.has_expression()) {
    auto E = s.expression().compile().evaluator();
    auto membership =
        tf::csg::graph::evaluate_per_domain(graph.inclusion(), E);
    return tf::csg::graph::compute_chosen_sides(graph.descriptor(), membership,
                                                s.kind());
  }
  tf::buffer<std::int8_t> forward;
  forward.allocate(std::size_t(graph.labels().n_components()));
  tf::parallel_fill(forward, std::int8_t(1));
  return forward;
}

} // namespace tf::csg::graph
