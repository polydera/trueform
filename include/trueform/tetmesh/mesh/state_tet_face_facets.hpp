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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./tet_face_map_workspace.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <utility>

namespace tf::tetmesh {

/// Block `f` of `face_facets` holds, ascending, the facets the input face
/// `f` lies in: the facets of the entrance triangles `parents` assigns it,
/// `facets` the offsets of each facet's run of triangles, the facets' own
/// order where they are published densely. A face no
/// crossing cut lies in one facet. The records are sorted on the tables of
/// `work`.
template <typename Index>
auto state_tet_face_facets(std::size_t n_faces, const tf::buffer<Index> &facets,
                           const tf::buffer<Index> &parents,
                           tet_face_map_workspace<Index> &work,
                           tf::offset_block_buffer<Index, Index> &face_facets)
    -> void {
  using record_t = tet_face_facet<Index>;
  auto &records = work.face_facets;
  records.allocate(parents.size());
  tf::parallel_for_each(
      tf::make_sequence_range(facets.size() ? facets.size() - 1 : 0),
      [&records, &facets, &parents](std::size_t f) {
        for (auto t = std::size_t(facets[f]); t < std::size_t(facets[f + 1]);
             ++t)
          records[t] = {parents[t], Index(f)};
      },
      tf::checked);
  tbb::parallel_sort(
      records.begin(), records.end(), [](const record_t &x, const record_t &y) {
        return x.face < y.face || (x.face == y.face && x.facet < y.facet);
      });
  // in place after the sort, one pass of compares and moves, which an
  // ordered parallel copy of the distinct records does not beat
  records.erase_till_end(std::unique(
      records.begin(), records.end(), [](const record_t &x, const record_t &y) {
        return x.face == y.face && x.facet == y.facet;
      }));
  face_facets.data_buffer().clear();
  tf::generate_offset_blocks(
      tf::make_sequence_range(n_faces), face_facets.offsets_buffer(),
      face_facets.data_buffer(),
      [&records](std::size_t f, auto &out) {
        auto at = std::lower_bound(records.begin(), records.end(), Index(f),
                                   [](const record_t &record, Index face) {
                                     return record.face < face;
                                   });
        for (; at != records.end() && at->face == Index(f); ++at)
          out.push_back(at->facet);
      },
      tf::checked);
}

} // namespace tf::tetmesh
