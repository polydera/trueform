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
#include "../../core/config/verify.hpp"
#include "../cholesky_pattern.hpp"
#include "../cholesky_scratch.hpp"
#include "../symmetric_lower.hpp"
#include "./count_cholesky_columns.hpp"
#include "./make_elimination_tree.hpp"
#include "./make_permuted_lower.hpp"
#include "./make_supernode_rows.hpp"
#include "./make_supernodes.hpp"
#include "./partition_elimination_tree.hpp"
#include "./postorder_tree.hpp"
#include "tbb/task_arena.h"
#include <cstddef>
#include <limits>

namespace tf::linalg {

/// Read the structure of the Cholesky factor of `matrix` eliminated in
/// `elimination` (new to old), over the graph `pattern` already holds, as
/// @ref tf::linalg::make_symmetric_adjacency states it.
///
/// The order is composed with a postorder of its elimination tree, which
/// changes neither fill nor work. The columns are counted, cut into
/// supernodes (@ref tf::linalg::make_supernodes) with their rows and blocks,
/// and the supernodal tree is cut into
/// @ref tf::linalg::cholesky_blocks_per_worker blocks per worker the arena
/// has (@ref tf::linalg::partition_elimination_tree), each supernode
/// weighing the sum of its columns' squared row counts. Returns false where
/// the index type cannot name the factor's values.
template <typename Index>
auto analyze_cholesky_order(const tf::symmetric_lower<Index> &matrix,
                            const tf::buffer<Index> &elimination,
                            tf::cholesky_scratch<Index> &scratch,
                            tf::cholesky_pattern<Index> &pattern) -> bool {
  const std::size_t n = matrix.size();
  pattern.rank.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    pattern.rank[std::size_t(elimination[k])] = Index(k);
  tf::linalg::make_elimination_tree(
      pattern.adjacency_offsets, pattern.adjacency, elimination, pattern.rank,
      scratch.stamp, scratch.unordered_parent);
  tf::linalg::postorder_tree(scratch.unordered_parent, scratch.children,
                             scratch.siblings, scratch.stack, scratch.post);

  // The postorder relabels the tree it was read from, so the composed order
  // takes its tree by relabelling rather than by another walk.
  auto &position = scratch.stamp;
  auto &parent = scratch.parent;
  pattern.order.allocate(n);
  parent.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t before = std::size_t(scratch.post[k]);
    pattern.order[k] = elimination[before];
    pattern.rank[std::size_t(pattern.order[k])] = Index(k);
    position[before] = Index(k);
  }
  for (std::size_t k = 0; k < n; ++k) {
    const Index up = scratch.unordered_parent[std::size_t(scratch.post[k])];
    parent[k] = up == Index(-1) ? up : position[std::size_t(up)];
  }

  tf::linalg::make_permuted_lower(
      matrix, pattern.adjacency_offsets, pattern.adjacency,
      pattern.adjacency_entries, pattern.order, pattern.rank,
      pattern.lower_offsets, pattern.lower_rows, pattern.lower_slot);
  tf::linalg::count_cholesky_columns(
      pattern.lower_offsets, pattern.lower_rows, parent, scratch.children,
      scratch.siblings, scratch.stack, scratch.stamp, scratch.counts);
  tf::linalg::make_supernodes(parent, scratch.counts, scratch.owner,
                              scratch.fundamental, scratch.start, scratch.into,
                              scratch.zeros, pattern.supernode_offsets);
  tf::linalg::make_supernode_rows(
      pattern.supernode_offsets, parent, scratch.counts, pattern.lower_offsets,
      pattern.lower_rows, scratch.owner, scratch.children, scratch.siblings,
      scratch.stamp, pattern.supernode_parent, pattern.row_offsets,
      pattern.rows);

  const std::size_t supernodes = pattern.number_of_supernodes();
  auto &work = scratch.work;
  work.allocate(supernodes);
  pattern.value_offsets.allocate(supernodes + 1);
  std::size_t total = 0;
  for (std::size_t s = 0; s < supernodes; ++s) {
    pattern.value_offsets[s] = Index(total);
    const std::size_t k = std::size_t(pattern.supernode_offsets[s + 1] -
                                      pattern.supernode_offsets[s]);
    const std::size_t m =
        std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
    total += m * k;
    if (total > std::size_t(std::numeric_limits<Index>::max()))
      return false;
    double weight = 0.0;
    for (std::size_t t = 0; t < k; ++t)
      weight += double(m - t) * double(m - t);
    work[s] = weight;
  }
  pattern.value_offsets[supernodes] = Index(total);
  tf::linalg::partition_elimination_tree(
      pattern.supernode_parent,
      tf::linalg::cholesky_blocks_per_worker *
          std::size_t(tbb::this_task_arena::max_concurrency()),
      work, scratch.owner, scratch.stack, pattern.schedule,
      pattern.block_offsets);

  if constexpr (tf::core::verification) {
    auto &size = scratch.owner;
    size.allocate_and_initialize(n, Index(1));
    for (std::size_t k = 0; k < n; ++k) {
      tf::core::verify(
          std::size_t(pattern.rank[std::size_t(pattern.order[k])]) == k,
          "a pattern's order and rank are inverse permutations");
      const Index up = parent[k];
      tf::core::verify(up == Index(-1) || std::size_t(up) > k,
                       "an elimination tree's parent follows its child");
      if (up != Index(-1))
        size[std::size_t(up)] = Index(size[std::size_t(up)] + size[k]);
    }
    for (std::size_t k = 0; k < n; ++k) {
      const Index up = parent[k];
      if (up != Index(-1))
        tf::core::verify(std::size_t(up) <
                             k + std::size_t(size[std::size_t(up)]),
                         "a postordered subtree is a contiguous run");
      const std::size_t first = std::size_t(pattern.lower_offsets[k]);
      const std::size_t last = std::size_t(pattern.lower_offsets[k + 1]);
      tf::core::verify(last > first &&
                           std::size_t(pattern.lower_rows[first]) == k,
                       "a lower column holds its diagonal first");
      for (std::size_t at = first + 1; at < last; ++at)
        tf::core::verify(std::size_t(pattern.lower_rows[at]) > k,
                         "a lower column's rows lie below its diagonal");
    }

    tf::core::verify(pattern.supernode_offsets[0] == Index(0) &&
                         std::size_t(pattern.supernode_offsets[supernodes]) ==
                             n,
                     "the supernodes cover the columns");
    for (std::size_t s = 0; s < supernodes; ++s) {
      const std::size_t first = std::size_t(pattern.supernode_offsets[s]);
      const std::size_t last = std::size_t(pattern.supernode_offsets[s + 1]);
      tf::core::verify(first < last, "a supernode holds a column");
      tf::core::verify(first + std::size_t(size[last - 1]) >= last,
                       "a supernode lies in its last column's subtree");
      const Index *rows =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[s]);
      const std::size_t m =
          std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
      for (std::size_t t = 0; t < last - first; ++t) {
        tf::core::verify(std::size_t(rows[t]) == first + t,
                         "a supernode's rows start with its own columns");
        tf::core::verify(std::size_t(scratch.counts[first + t]) <= m - t,
                         "a supernode's block holds every row its columns "
                         "count");
      }
      for (std::size_t p = 1; p < m; ++p)
        tf::core::verify(rows[p - 1] < rows[p], "a supernode's rows ascend");
      const Index up = pattern.supernode_parent[s];
      tf::core::verify((up == Index(-1)) == (m == last - first),
                       "a supernode updates a parent exactly where rows lie "
                       "below its columns");
      if (up == Index(-1))
        continue;
      tf::core::verify(std::size_t(up) > s, "a supernode's parent follows it");
      const Index *above =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[up]);
      const Index *above_end =
          pattern.rows.begin() + std::size_t(pattern.row_offsets[up + 1]);
      tf::core::verify(rows[last - first] >=
                               pattern.supernode_offsets[std::size_t(up)] &&
                           rows[last - first] <
                               pattern.supernode_offsets[std::size_t(up) + 1],
                       "a supernode's first row below it is its parent's");
      for (std::size_t p = last - first; p < m; ++p) {
        while (above != above_end && *above < rows[p])
          ++above;
        tf::core::verify(above != above_end && *above == rows[p],
                         "a supernode's update lands in its parent's rows");
      }
    }

    auto &block_of = scratch.stamp;
    block_of.allocate_and_initialize(supernodes, Index(-1));
    for (std::size_t block = 0; block < pattern.number_of_blocks(); ++block)
      for (std::size_t slot = std::size_t(pattern.block_offsets[block]);
           slot < std::size_t(pattern.block_offsets[block + 1]); ++slot) {
        const std::size_t s = std::size_t(pattern.schedule[slot]);
        tf::core::verify(
            s == std::size_t(pattern.schedule[std::size_t(
                     pattern.block_offsets[block])]) +
                     (slot - std::size_t(pattern.block_offsets[block])),
            "a block is a contiguous run");
        block_of[s] = Index(block);
      }
    for (std::size_t slot = pattern.frontier_first(); slot < supernodes;
         ++slot) {
      tf::core::verify(slot == pattern.frontier_first() ||
                           pattern.schedule[slot - 1] < pattern.schedule[slot],
                       "the frontier ascends");
      block_of[std::size_t(pattern.schedule[slot])] = Index(-2);
    }
    for (std::size_t s = 0; s < supernodes; ++s) {
      tf::core::verify(block_of[s] != Index(-1),
                       "the schedule takes every supernode once");
      const Index up = pattern.supernode_parent[s];
      tf::core::verify(up == Index(-1) ||
                           block_of[std::size_t(up)] == Index(-2) ||
                           block_of[std::size_t(up)] == block_of[s],
                       "a block holds whole subtrees under the frontier");
    }
  }
  return true;
}

} // namespace tf::linalg
