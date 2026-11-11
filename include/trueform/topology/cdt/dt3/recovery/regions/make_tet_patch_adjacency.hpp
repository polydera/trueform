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
#include "../../../../../core/algorithm/generic_generate.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include "../../../../../core/offset_block_buffer.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "./tet_patch_workspace.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Per patch below `n_patches` of `patch_of_triangle`, the patches it meets
/// across an edge `applier(t, f)` names, ascending, as its block of
/// `adjacency`: the patch graph the frontier's middle rung is grown on
/// (@ref claim_tet_patch_ranges, @ref join_tet_patch_pockets), symmetric
/// where the applier is. `work.pairs` is
/// the pass's scratch.
template <typename Applier, typename Index>
auto make_tet_patch_adjacency(const Applier &applier,
                              const tf::buffer<Index> &patch_of_triangle,
                              Index n_patches,
                              tf::offset_block_buffer<Index, Index> &adjacency,
                              tet_patch_workspace<Index> &work) -> void {
  auto &pairs = work.pairs;
  pairs.clear();
  tf::generic_generate(
      tf::make_sequence_range(patch_of_triangle.size()), pairs,
      [&applier, &patch_of_triangle](std::size_t t, auto &out) {
        const Index p = patch_of_triangle[t];
        if (p < 0)
          return;
        applier(Index(t), [&out, &patch_of_triangle, p](Index u) {
          const Index q = patch_of_triangle[std::size_t(u)];
          if (q >= 0 && q != p)
            out.push_back(std::array<Index, 2>{p, q});
        });
      },
      tf::checked);
  tbb::parallel_sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(pairs.begin(), pairs.end()));
  auto &offsets = adjacency.offsets_buffer();
  auto &data = adjacency.data_buffer();
  offsets.allocate(std::size_t(n_patches) + 1);
  data.allocate(pairs.size());
  std::size_t at = 0;
  for (std::size_t p = 0; p < std::size_t(n_patches); ++p) {
    offsets[p] = Index(at);
    while (at < pairs.size() && std::size_t(pairs[at][0]) == p) {
      data[at] = pairs[at][1];
      ++at;
    }
  }
  offsets[std::size_t(n_patches)] = Index(at);
}

} // namespace tf::topology::cdt::dt3
