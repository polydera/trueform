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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/slide_range.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"
#include "./make_labels.hpp"
#include "./partition_ids.hpp"
#include "./partition_labels.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/task_group.h"

namespace tf::cut {

/// Build partition_ids from partition_labels.
/// Maps each component to the list of polygon/cut-face indices belonging
/// to it via counting sort.
template <typename Index, typename LabelType>
auto make_partition_ids(const tf::cut::partition_labels<LabelType> &labels) {
  tf::cut::partition_ids<Index> ids;
  auto make_f = [&, n_components = labels.n_components](
                    tf::offset_block_buffer<Index, Index> &b,
                    const auto &labels) {
    b.offsets_buffer().allocate(n_components + 1);
    tf::parallel_fill(b.offsets_buffer(), 0);
    Index empty_count = 0;
    for (auto l : labels)
      if (l != -1)
        b.offsets_buffer()[l]++;
      else
        empty_count++;
    for (auto b : tf::make_slide_range<2>(b.offsets_buffer()))
      b[1] += b[0];
    b.data_buffer().allocate(labels.size() - empty_count);
    for (auto [i, l] : tf::enumerate(labels))
      if (l != -1)
        b.data_buffer()[--b.offsets_buffer()[l]] = i;
  };
  tbb::parallel_invoke([&] { make_f(ids.polygons, labels.polygon_labels); },
                       [&] { make_f(ids.cut_faces, labels.cut_labels); });
  return ids;
}

/// Build partition_ids from a vector of partition_labels.
template <typename Index, typename LabelType>
auto make_partition_ids(
    const tf::small_vector<tf::cut::partition_labels<LabelType>, 4> &pals) {
  auto n = pals.size();
  tf::small_vector<tf::cut::partition_ids<Index>, 4> result;
  result.resize(n);
  tbb::task_group tg;
  for (std::size_t t = 0; t < n; ++t)
    tg.run([&, t] { result[t] = make_partition_ids<Index>(pals[t]); });
  tg.wait();
  return result;
}

/// Build partition_ids for a 2-mesh pair.
template <typename LabelType, typename Index, typename Policy0,
          typename Policy1>
auto make_partition_ids(const tf::polygons<Policy0> &form0,
                        const tf::polygons<Policy1> &form1,
                        const tf::face_cuts<Index> &fc,
                        const tf::cut_graph<Index> &cg) {
  auto pals = make_partition_labels<LabelType>(form0, form1, fc, cg);

  tf::small_vector<tf::cut::partition_ids<Index>, 4> result;
  result.resize(2);
  tbb::parallel_invoke(
      [&] { result[0] = make_partition_ids<Index>(pals[0]); },
      [&] { result[1] = make_partition_ids<Index>(pals[1]); });
  return result;
}

/// Build partition_ids for N meshes.
template <typename LabelType, typename Index, typename Iterator, std::size_t N>
auto make_partition_ids(tf::range<Iterator, N> forms,
                        const tf::face_cuts<Index> &fc,
                        const tf::cut_graph<Index> &cg) {
  auto pals = make_partition_labels<LabelType>(forms, fc, cg);
  auto n_tags = forms.size();

  tf::small_vector<tf::cut::partition_ids<Index>, 4> result;
  result.resize(n_tags);
  tbb::task_group tg;
  for (std::size_t t = 0; t < n_tags; ++t)
    tg.run([&, t] { result[t] = make_partition_ids<Index>(pals[t]); });
  tg.wait();
  return result;
}

} // namespace tf::cut
