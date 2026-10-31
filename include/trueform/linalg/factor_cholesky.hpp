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
#include "../core/algorithm/parallel_for.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/config/verify.hpp"
#include "../core/views/sequence_range.hpp"
#include "./cholesky/cholesky_front_scratch.hpp"
#include "./cholesky/factor_supernode_run.hpp"
#include "./cholesky_factor.hpp"
#include "./cholesky_pattern.hpp"
#include "./cholesky_scratch.hpp"
#include <algorithm>
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief Factor one value set of the pattern @ref tf::analyze_cholesky
/// read, and state whether the matrix is positive definite.
///
/// `values` holds one value per entry of the pattern's input, in its order;
/// they reach the permuted carrier through its tickets. A matrix that is not
/// positive definite, or a value that is not finite, refuses, and the factor
/// then states nothing.
///
/// The factor is multifrontal over the supernodes: each supernode gathers its
/// lower columns and its children's updates into a dense front, factors its
/// own columns there with dense kernels, and passes the trailing block up as
/// its parent's update. The subtrees of the pattern's schedule factor in
/// parallel, each on its own stack, and the frontier above them after, with
/// the tiles of each front in parallel.
///
/// Every entry takes its arithmetic in one order the pattern fixes: a front
/// adds its children in ascending order, and the dense kernels cut their
/// work into tiles of constant size whose sums run in a fixed order. So the
/// factor is the same bits at every thread count, every partition and every
/// run. It is not the bits of a column-by-column factor of the same matrix,
/// which sums in another order; the two agree to the condition times the
/// unit roundoff. No pivot is compared with anything but zero: a definite
/// system may carry pivots many orders of magnitude below their diagonal.
template <typename Index>
auto factor_cholesky(const tf::cholesky_pattern<Index> &pattern,
                     const tf::buffer<double> &values,
                     tf::cholesky_scratch<Index> &scratch,
                     tf::cholesky_factor &factor) -> bool {
  auto &lower = scratch.lower_values;
  lower.allocate(values.size());
  tf::parallel_for(
      std::size_t(0), values.size(),
      [&](std::size_t first, std::size_t last) {
        for (std::size_t e = first; e < last; ++e)
          lower[std::size_t(pattern.lower_slot[e])] = values[e];
      },
      tf::checked);
  factor.values.allocate(
      std::size_t(pattern.value_offsets[pattern.number_of_supernodes()]));

  // A block hands its root's update to the frontier through its own run of
  // `handoff`; a root of the whole tree hands nothing.
  const std::size_t blocks = pattern.number_of_blocks();
  auto root_of = [&](std::size_t block) {
    return std::size_t(
        pattern.schedule[std::size_t(pattern.block_offsets[block + 1]) - 1]);
  };
  auto update_size = [&](std::size_t s) -> std::size_t {
    if (pattern.supernode_parent[s] == Index(-1))
      return 0;
    const std::size_t rest =
        std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]) -
        std::size_t(pattern.supernode_offsets[s + 1] -
                    pattern.supernode_offsets[s]);
    return rest * (rest + 1) / 2;
  };
  auto &handoff_offsets = scratch.handoff_offsets;
  handoff_offsets.allocate(blocks + 1);
  handoff_offsets[0] = 0;
  for (std::size_t block = 0; block < blocks; ++block)
    handoff_offsets[block + 1] =
        handoff_offsets[block] + update_size(root_of(block));
  scratch.handoff.allocate(handoff_offsets[blocks]);
  scratch.refused.allocate_and_initialize(blocks, char(0));
  tf::parallel_for_each(
      tf::make_sequence_range(blocks),
      [&](std::size_t block, tf::linalg::cholesky_front_scratch<Index> &local) {
        if (!tf::linalg::factor_supernode_run(
                pattern, lower, std::size_t(pattern.block_offsets[block]),
                std::size_t(pattern.block_offsets[block + 1]),
                [](std::size_t) {}, local, factor)) {
          scratch.refused[block] = char(1);
          return;
        }
        if constexpr (tf::core::verification)
          tf::core::verify(local.pending.size() ==
                               std::size_t(update_size(root_of(block)) > 0),
                           "a block owes only its root's update");
        std::copy(local.stack.begin(),
                  local.stack.begin() +
                      std::ptrdiff_t(handoff_offsets[block + 1] -
                                     handoff_offsets[block]),
                  scratch.handoff.begin() + handoff_offsets[block]);
      },
      tf::linalg::cholesky_front_scratch<Index>{});
  for (std::size_t block = 0; block < blocks; ++block)
    if (scratch.refused[block])
      return false;

  auto &front = scratch.front;
  std::size_t fed = 0;
  return tf::linalg::factor_supernode_run(
      pattern, lower, pattern.frontier_first(), pattern.number_of_supernodes(),
      [&](std::size_t s) {
        for (; fed < blocks && root_of(fed) < s; ++fed) {
          const std::size_t size =
              handoff_offsets[fed + 1] - handoff_offsets[fed];
          if (!size)
            continue;
          front.pending.push_back(Index(root_of(fed)));
          front.pending_offsets.push_back(front.stack.size());
          front.stack.reallocate(front.stack.size() + size);
          std::copy(scratch.handoff.begin() + handoff_offsets[fed],
                    scratch.handoff.begin() + handoff_offsets[fed + 1],
                    front.stack.end() - std::ptrdiff_t(size));
        }
      },
      front, factor);
}

} // namespace tf
