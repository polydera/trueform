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
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "./face_descriptor.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// A member's winding in its plane's projection, read off the ORIGINAL
/// face: the loop is that face subdivided, so the two windings are one
/// fact, and the face's is independent of every weld and mint the
/// loop's would have to be measured before.
///
/// `plane_of` is the carrier lookup, whatever states it — a pooled graph's
/// dense table, or a world whose face IS its plane.
template <typename Index, typename Int, typename PlaneOf, typename ApplyToFace,
          typename GetPoint>
auto build_plane_orientations(
    const tf::buffer<face_descriptor<Index>> &descriptors,
    const PlaneOf &plane_of,
    const tf::buffer<tf::exact::plane_frame<Int>> &frames,
    const ApplyToFace &apply_to_face, const GetPoint &get_point,
    tf::buffer<std::int8_t> &face_orientation) -> void {
  face_orientation.allocate(descriptors.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(descriptors.size())),
      [&](Index f) {
        const auto tag = std::int16_t(descriptors[std::size_t(f)].tag);
        const auto object = descriptors[std::size_t(f)].object;
        const auto &frame = frames[std::size_t(plane_of[std::size_t(f)])];
        apply_to_face(tag, object, [&](const auto &face) {
          face_orientation[std::size_t(f)] = tf::exact::plane_frame_winding(
              frame, face,
              [&](auto corner) { return get_point(tag, Index(corner)); });
        });
      },
      tf::checked);
}

} // namespace tf::intersect::graph
