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
#include "../core/none.hpp"
#include "./edge_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create edge membership from segments.
///
/// Convenience function that builds and returns an edge_membership structure.
/// The index type is automatically deduced from the segments unless specified.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The segments policy type.
/// @param segments The segments range.
/// @param eo The edge orientation mode (default: bidirectional).
/// @return An edge_membership structure mapping vertices to edges.
template <typename Index = tf::none_t, typename Policy>
auto make_edge_membership(
    const tf::segments<Policy> &segments,
    tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_edge_membership<std::decay_t<decltype(segments.edges()[0][0])>>(
        segments, eo);
  } else {
    tf::edge_membership<Index> em;
    em.build(segments, eo);
    return em;
  }
}

} // namespace tf
