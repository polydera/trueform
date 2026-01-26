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
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "./by_ids.hpp"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

namespace tf {

/// @cond INTERNAL
namespace reindex {
template <typename Index, typename T, typename Range>
auto split_into_components(const T &t, const Range &labels) {
  tf::buffer<Index> ids;
  ids.allocate(labels.size());
  tf::parallel_iota(ids, Index(0));
  tbb::parallel_sort(ids.begin(), ids.end(),
                     [&](auto i0, auto i1) { return labels[i0] < labels[i1]; });
  tf::buffer<Index> offsets;
  tf::compute_offsets(
      ids, std::back_inserter(offsets), Index(0),
      [&](auto i0, auto i1) { return labels[i0] == labels[i1]; });
  auto ids_r = tf::make_offset_block_range(offsets, ids);
  using label_t = std::decay_t<decltype(labels[0])>;
  using res_t = decltype(tf::reindexed_by_ids<Index>(t, ids_r.front()));
  std::vector<res_t> out;
  std::vector<label_t> l_out;
  out.resize(ids_r.size());
  l_out.resize(ids_r.size());
  tbb::task_group tg;
  for (auto &&[ids, label, res] : tf::zip(ids_r, l_out, out))
    tg.run(
        [&res = res, &t, ids = tf::make_range(ids), &labels, &label = label] {
          label = labels[ids[0]];
          res = tf::reindexed_by_ids<Index>(t, ids);
        });
  tg.wait();
  return std::make_pair(out, std::move(l_out));
}
} // namespace reindex
/// @endcond

/// @ingroup reindex
/// @brief Split polygons into labeled components.
///
/// Groups faces by label and returns separate geometry for each.
/// Uses @ref tf::reindexed_by_ids internally.
///
/// @tparam Policy The policy type of the polygons.
/// @tparam Range The label range type.
/// @param polygons The input @ref tf::polygons.
/// @param labels Per-face labels (same size as faces()).
/// @return Pair of (vector of @ref tf::polygons_buffer, vector of labels).
template <typename Policy, typename Range>
auto split_into_components(const tf::polygons<Policy> &polygons,
                           const Range &labels) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  return reindex::split_into_components<Index>(polygons, labels);
}

/// @ingroup reindex
/// @brief Split segments into labeled components.
///
/// Groups edges by label and returns separate geometry for each.
/// Uses @ref tf::reindexed_by_ids internally.
///
/// @tparam Policy The policy type of the segments.
/// @tparam Range The label range type.
/// @param segments The input @ref tf::segments.
/// @param labels Per-edge labels (same size as edges()).
/// @return Pair of (vector of @ref tf::segments_buffer, vector of labels).
template <typename Policy, typename Range>
auto split_into_components(const tf::segments<Policy> &segments,
                           const Range &labels) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  return reindex::split_into_components<Index>(segments, labels);
}
} // namespace tf
