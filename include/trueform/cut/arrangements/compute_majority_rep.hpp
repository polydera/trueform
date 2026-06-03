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
#include "../../core/buffer.hpp"
#include "../../topology/domains/compute_majority_rep.hpp"

namespace tf::cut {

/// @ingroup cut
/// @brief Per NM-edge set (edges sharing the same incident component-label
/// multiset), find the majority canonical radial perm and emit ONE
/// representative edge id per set.
///
/// Graph-side alias for @ref tf::topology::domains::compute_majority_rep:
/// the algorithm depends only on `is_valid + edge_path_id +
/// id_sorted_view + labels_view`, all of which are Index-block ranges
/// produced identically on both the materialised-polygons and
/// arrangement-graph paths. Single source of truth, reachable from
/// `tf::cut::` for API consistency.
template <typename Index, typename IdSortedView, typename LabelsView>
auto compute_majority_rep(Index n_edges, const tf::buffer<char> &is_valid,
                          const tf::buffer<Index> &edge_path_id,
                          const IdSortedView &id_sorted_view,
                          const LabelsView &labels_view) -> tf::buffer<Index> {
  return tf::topology::domains::compute_majority_rep(
      n_edges, is_valid, edge_path_id, id_sorted_view, labels_view);
}

} // namespace tf::cut
