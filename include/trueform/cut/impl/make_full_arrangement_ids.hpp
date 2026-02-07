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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "./polygon_arrangement_ids.hpp"

namespace tf::cut {

template <typename Index, typename Range>
auto make_full_arrangement_ids(std::size_t n_polygons, std::size_t n_cut_faces,
                               const Range &descriptors) {
  tf::cut::polygon_arrangement_ids<Index> pai;

  tf::buffer<bool> cut_mask;
  cut_mask.allocate(n_polygons);
  tf::parallel_fill(cut_mask, false);
  tf::parallel_for_each(descriptors,
                        [&](auto d) { cut_mask[d.object] = true; }, tf::checked);

  std::size_t n_uncut = 0;
  for (auto m : cut_mask)
    if (!m)
      ++n_uncut;

  pai.polygons.offsets_buffer().allocate(2);
  pai.polygons.offsets_buffer()[0] = 0;
  pai.polygons.offsets_buffer()[1] = static_cast<Index>(n_uncut);
  pai.polygons.data_buffer().allocate(n_uncut);

  Index idx = 0;
  for (Index i = 0; i < static_cast<Index>(n_polygons); ++i)
    if (!cut_mask[i])
      pai.polygons.data_buffer()[idx++] = i;

  pai.cut_faces.offsets_buffer().allocate(2);
  pai.cut_faces.offsets_buffer()[0] = 0;
  pai.cut_faces.offsets_buffer()[1] = static_cast<Index>(n_cut_faces);
  pai.cut_faces.data_buffer().allocate(n_cut_faces);
  tf::parallel_iota(pai.cut_faces.data_buffer(), Index(0));

  return pai;
}

} // namespace tf::cut
