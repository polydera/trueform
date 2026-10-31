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
#include "../../core/blocked_buffer.hpp"
#include "../../core/resolved_output_real.hpp"
#include "../../topology/domain_config.hpp"
#include "../csg_graph.hpp"
#include "./compute_domain_membership.hpp"
#include "./domain_membership_for.hpp"
#include "./make_csg_domains.hpp"
#include "./make_open_fragment_partition.hpp"
#include <cstddef>
#include <tuple>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The open fragments of a built graph, one mesh per form, through
///        the domain emitter. Under `WantPointMap` the inclusion matrix holds
///        no row: a fragment bounds no domain.
template <typename OutputCoordinateType, bool WantLabels = false,
          bool WantPointMap = false, typename Policy, typename Int,
          template <typename, typename> class Arrangement>
auto open_fragments_for(const tf::csg_graph<Policy, Int, Arrangement> &graph,
                        tf::domain_config config) {
  using graph_t = tf::csg_graph<Policy, Int, Arrangement>;
  using Index = typename graph_t::index_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             typename graph_t::input_real_type>;
  const Index n_tags = Index(graph.arrangement().n_tags());
  // Nothing arranged is no component: every form's fragment is empty.
  auto membership = graph.descriptor().n_domains == Index(0)
                        ? tf::csg::graph::domain_membership<Index>{}
                        : tf::csg::graph::domain_membership_for(
                              graph, config, [](const auto &) { return true; });
  auto part = tf::csg::graph::make_open_fragment_partition(
      membership, graph.descriptor().tag_of_component, n_tags);
  auto result =
      tf::csg::graph::make_csg_domains<RealOut, graph_t::face_static_size,
                                       WantLabels, WantPointMap, Index>(
          graph.arrangement(), graph.labels(), part);
  if constexpr (WantPointMap)
    std::get<2>(result).inclusion =
        tf::blocked_buffer<bool, tf::dynamic_size>(std::size_t(n_tags));
  return result;
}

} // namespace tf::csg::graph
