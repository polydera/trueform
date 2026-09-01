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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_edge_def.hpp"
#include "tbb/parallel_sort.h"
#include <cstddef>
#include <iterator>

namespace tf::intersect::graph {

/// Canonical identity IS the record's leading key, so the sort that
/// proves it is the sort that groups the table: the spans are the
/// group -> instances lookup, and the group id goes into each row.
template <typename Index>
auto canonicalize_plane_edge_defs(tf::buffer<plane_edge_def<Index>> &edge_defs,
                                  tf::buffer<Index> &edge_defs_offsets,
                                  Index &n_canon,
                                  tf::buffer<Index> &emission_to_canon) -> void {
  using def_t = plane_edge_def<Index>;
  n_canon = 0;
  edge_defs_offsets.clear();
  if (edge_defs.size() == 0)
    return;
  tbb::parallel_sort(edge_defs.begin(), edge_defs.end(),
                     [](const def_t &x, const def_t &y) {
                       return plane_def_key_less(x, y) ||
                              (!plane_def_key_less(y, x) && x.id < y.id);
                     });
  edge_defs_offsets.reserve(edge_defs.size() + 1);
  tf::compute_offsets(edge_defs, std::back_inserter(edge_defs_offsets),
                      Index(0), [](const def_t &x, const def_t &y) {
                        return !plane_def_key_less(x, y) &&
                               !plane_def_key_less(y, x);
                      });
  n_canon = Index(edge_defs_offsets.size()) - 1;
  emission_to_canon.allocate(edge_defs.size());
  tf::parallel_for_each(
      tf::make_sequence_range(n_canon),
      [&](Index g) {
        const auto b = std::size_t(edge_defs_offsets[std::size_t(g)]);
        const auto e = std::size_t(edge_defs_offsets[std::size_t(g) + 1]);
        for (auto i = b; i < e; ++i) {
          emission_to_canon[std::size_t(edge_defs[i].id)] = Index(i);
          edge_defs[i].id = g;
        }
      },
      tf::checked);
}

} // namespace tf::intersect::graph
