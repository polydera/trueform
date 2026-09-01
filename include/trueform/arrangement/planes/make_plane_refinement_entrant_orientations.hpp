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
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/plane_frame_winding.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// State each newly reached source face's winding in its exact plane frame.
template <typename Index, typename Descriptors, typename Frames,
          typename GetPoint, typename ApplyToForm>
auto make_plane_refinement_entrant_orientations(
    const Descriptors &descriptors, const Frames &frames,
    const GetPoint &get_point, const ApplyToForm &apply_to_form,
    tf::buffer<std::int8_t> &orientations) -> void {
  orientations.allocate(descriptors.size());
  tf::parallel_for_each(
      tf::make_sequence_range(descriptors.size()),
      [&](std::size_t row) {
        const auto descriptor = descriptors[row];
        apply_to_form(Index(descriptor.tag), [&](const auto &form) {
          orientations[row] = tf::exact::plane_frame_winding(
              frames[row], form.faces()[descriptor.object],
              [&](auto corner) {
                return get_point(std::int16_t(descriptor.tag), Index(corner));
              });
        });
      },
      tf::checked);
}

} // namespace tf::arrangement
