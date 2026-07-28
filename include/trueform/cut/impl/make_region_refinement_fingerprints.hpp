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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./region_refinement_fingerprint.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto make_region_refinement_fingerprints(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::offset_block_buffer<Index, Index> &region_steiners,
    tf::buffer<std::array<Index, 2>> &fingerprints) -> void {
  auto descriptors = regions.descriptors();
  auto loops = regions.loops();
  fingerprints.allocate(std::size_t(loops.size()));
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(loops.size())),
      [&](std::size_t region) {
        fingerprints[region] =
            descriptors[Index(region)].tag == Index(-1)
                ? std::array<Index, 2>{0, 0}
                : tf::cut::region_refinement_fingerprint(
                      regions, Index(region), n_tags, splits,
                      region_steiners);
      });
}

} // namespace tf::cut
