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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"

#include <cstddef>

namespace tf::arrangement {

/// Build refinement's fixed dense source-face exclusion mask.
///
/// `world.descriptors()` is the complete immutable LA carrier: its cut
/// prefix followed by its LA entrant suffix. Source forms own the per-tag face
/// extents, so this mask is the one authority on which source faces a world
/// holds — shared by ring propagation, refinement entrant discovery, and the
/// wave entrance's own answered set
/// (@ref tf::arrangement::plane_wave_answered), which seeds from it.
template <typename Index, typename World, typename ApplyToForm>
auto make_plane_refinement_cut_mask(const World &world, Index n_tags,
                                    const ApplyToForm &apply_to_form)
    -> tf::core::std_vector<tf::buffer<char>> {
  tf::core::std_vector<tf::buffer<char>> cut_mask{std::size_t(n_tags)};
  for (Index tag = 0; tag < n_tags; ++tag)
    apply_to_form(tag, [&cut_mask, tag](const auto &form) {
      auto &mask = cut_mask[std::size_t(tag)];
      mask.allocate(form.faces().size());
      tf::parallel_fill(mask, char(0));
    });
  for (const auto &descriptor : world.descriptors())
    cut_mask[std::size_t(descriptor.tag)][std::size_t(descriptor.object)] = 1;
  return cut_mask;
}

} // namespace tf::arrangement
