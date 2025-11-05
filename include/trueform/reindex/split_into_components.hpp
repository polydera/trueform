/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "./by_ids.hpp"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

namespace tf {
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
  using res_t = decltype(tf::reindexed_by_ids<Index>(t, ids_r.front()));
  std::vector<res_t> out;
  out.resize(ids_r.size());
  tbb::task_group tg;
  for (auto &&[ids, res] : tf::zip(ids_r, out))
    tg.run([&res = res, &t, ids = tf::make_range(ids)] {
      res = tf::reindexed_by_ids<Index>(t, ids);
    });
  tg.wait();
  return out;
}
} // namespace reindex
template <typename Policy, typename Range>
auto split_into_components(const tf::polygons<Policy> &polygons,
                           const Range &labels) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  return reindex::split_into_components<Index>(polygons, labels);
}

template <typename Policy, typename Range>
auto split_into_components(const tf::segments<Policy> &segments,
                           const Range &labels) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  return reindex::split_into_components<Index>(segments, labels);
}
} // namespace tf
