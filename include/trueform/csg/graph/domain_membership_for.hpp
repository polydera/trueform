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
#include "../../topology/domain_config.hpp"
#include "../csg_graph.hpp"
#include "./compute_domain_membership.hpp"
#include "./structural_membership.hpp"

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The coarse domain membership of a built graph under `config` and
///        `E`: the one site every domain read coarsens through.
/// @pre `graph.descriptor().n_domains > 0`.
template <typename Policy, typename Int,
          template <typename, typename> class Arrangement, typename Expr>
auto domain_membership_for(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                           tf::domain_config config, Expr E) {
  using Index = typename tf::csg_graph<Policy, Int, Arrangement>::index_type;
  return tf::csg::graph::compute_domain_membership(
      graph.descriptor(), graph.inclusion(),
      graph.labels().open_component_mask(), graph.domain_nesting_merges(),
      config, E,
      Index(tf::csg::graph::find_universe_domain(graph.domain_volumes())),
      graph.crossings(), graph.is_sheet());
}

} // namespace tf::csg::graph
