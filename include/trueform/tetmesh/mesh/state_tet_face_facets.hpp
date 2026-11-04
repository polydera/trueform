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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <utility>

namespace tf::tetmesh {

/// Block `f` of `face_facets` holds, ascending, the facets the input face
/// `f` lies in: the facets of the entrance triangles `parents` assigns it,
/// `facets` the offsets of each facet's run of triangles. A face no
/// crossing cut lies in one facet.
template <typename Index>
auto state_tet_face_facets(std::size_t n_faces, const tf::buffer<Index> &facets,
                           const tf::buffer<Index> &parents,
                           tf::offset_block_buffer<Index, Index> &face_facets)
    -> void {
  tf::buffer<std::pair<Index, Index>> records;
  records.allocate(parents.size());
  for (std::size_t f = 0; f + 1 < facets.size(); ++f)
    for (auto t = std::size_t(facets[f]); t < std::size_t(facets[f + 1]); ++t)
      records[t] = {parents[t], Index(f)};
  tbb::parallel_sort(records.begin(), records.end());
  records.erase_till_end(std::unique(records.begin(), records.end()));
  face_facets.data_buffer().clear();
  tf::generate_offset_blocks(
      tf::make_sequence_range(n_faces), face_facets.offsets_buffer(),
      face_facets.data_buffer(),
      [&records](std::size_t f, auto &out) {
        auto at = std::lower_bound(records.begin(), records.end(),
                                   std::make_pair(Index(f), Index(0)));
        for (; at != records.end() && at->first == Index(f); ++at)
          out.push_back(at->second);
      },
      tf::checked);
}

} // namespace tf::tetmesh
