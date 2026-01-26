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
#include "../../core/views/enumerate.hpp"
#include "../../core/views/slide_range.hpp"
#include "./polygon_arrangement_ids.hpp"
#include "./polygon_arrangement_labels.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {
template <typename Index, typename LabelType>
auto make_polygon_arrangement_ids(
    const tf::cut::polygon_arrangement_labels<LabelType> &labels) {
  tf::cut::polygon_arrangement_ids<Index> pai;
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
  tbb::parallel_invoke([&] { make_f(pai.polygons, labels.polygon_labels); },
                       [&] { make_f(pai.cut_faces, labels.cut_labels); });
  return pai;
}
} // namespace tf::cut
