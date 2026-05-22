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
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Per NM-edge set (edges sharing the same incident fragment-label
/// multiset), find the majority canonical radial perm and emit ONE
/// representative edge id per set.
///
/// Returns a compact buffer of unique rep edge ids (one per valid set).
/// Reps are in their post-canonicalize_nm_edges state (Class A/B,
/// internally consistent (face_block, axis) pair). Downstream
/// emit_domain_merges should iterate this buffer directly.
template <typename Index, typename IdSortedView, typename LabelsView>
auto compute_majority_rep(Index n_edges, const tf::buffer<char> &is_valid,
                          const tf::buffer<Index> &edge_path_id,
                          const IdSortedView &id_sorted_view,
                          const LabelsView &labels_view)
    -> tf::buffer<Index> {
  tf::buffer<Index> edge_order;
  edge_order.allocate(n_edges);
  tf::parallel_iota(edge_order, Index(0));
  tbb::parallel_sort(
      edge_order.begin(), edge_order.end(), [&](Index a, Index b) {
        if (is_valid[a] != is_valid[b])
          return is_valid[a] > is_valid[b];
        if (!is_valid[a])
          return a < b;
        if (edge_path_id[a] != edge_path_id[b])
          return edge_path_id[a] < edge_path_id[b];
        auto sa = id_sorted_view[a];
        auto sb = id_sorted_view[b];
        if (!std::equal(sa.begin(), sa.end(), sb.begin(), sb.end()))
          return std::lexicographical_compare(sa.begin(), sa.end(), sb.begin(),
                                              sb.end());
        auto pa = labels_view[a];
        auto pb = labels_view[b];
        return std::lexicographical_compare(pa.begin(), pa.end(), pb.begin(),
                                            pb.end());
      });

  tf::buffer<Index> set_offsets;
  tf::compute_offsets(edge_order, std::back_inserter(set_offsets), Index(0),
                      [&](Index a, Index b) {
                        if (is_valid[a] != is_valid[b])
                          return false;
                        if (!is_valid[a])
                          return true;
                        if (edge_path_id[a] != edge_path_id[b])
                          return false;
                        auto sa = id_sorted_view[a];
                        auto sb = id_sorted_view[b];
                        return std::equal(sa.begin(), sa.end(), sb.begin(),
                                          sb.end());
                      });
  auto sets = tf::make_offset_block_range(set_offsets, edge_order);

  tf::buffer<Index> reps;
  tf::generic_generate(sets, reps, [&](auto set, auto &out) {
    if (set.size() == 0)
      return;
    Index first = set.begin()[0];
    if (!is_valid[first])
      return;

    Index best_count = 1;
    Index best_rep = first;
    Index current_count = 1;
    Index current_rep = first;
    auto prev = first;
    for (std::size_t i = 1; i < set.size(); ++i) {
      Index k = set.begin()[i];
      auto pa = labels_view[prev];
      auto pk = labels_view[k];
      if (std::equal(pa.begin(), pa.end(), pk.begin(), pk.end())) {
        ++current_count;
      } else {
        if (current_count > best_count) {
          best_count = current_count;
          best_rep = current_rep;
        }
        current_count = 1;
        current_rep = k;
      }
      prev = k;
    }
    if (current_count > best_count) {
      best_count = current_count;
      best_rep = current_rep;
    }
    out.push_back(best_rep);
  });

  return reps;
}

} // namespace tf::topology::domains
