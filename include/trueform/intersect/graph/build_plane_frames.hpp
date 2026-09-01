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
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "./face_descriptor.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// Per plane: the projection its arrangement is computed in, read off
/// the member that named it.
template <typename Index, typename Int, typename PlaneFaces,
          typename ApplyToFace, typename GetPoint>
auto build_plane_frames(Index n_planes, const PlaneFaces &plane_faces,
                        const tf::buffer<face_descriptor<Index>> &descriptors,
                        const ApplyToFace &apply_to_face,
                        const GetPoint &get_point,
                        tf::buffer<tf::exact::plane_frame<Int>> &frames)
    -> void {
  frames.allocate(std::size_t(n_planes));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        const auto member = plane_faces[std::size_t(p)][0];
        const auto tag = std::int16_t(descriptors[std::size_t(member)].tag);
        const auto object = descriptors[std::size_t(member)].object;
        apply_to_face(tag, object, [&](const auto &face) {
          frames[std::size_t(p)] = tf::exact::make_supported_plane_frame<Int>(
              [&](const auto &consider) {
                for (const auto corner : face)
                  consider(get_point(tag, Index(corner)));
              });
        });
      },
      tf::checked);
}

} // namespace tf::intersect::graph
