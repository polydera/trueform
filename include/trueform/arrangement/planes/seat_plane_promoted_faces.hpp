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
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "../../intersect/graph/name_plane_carriers.hpp"
#include "../../intersect/graph/seat_named_planes.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// Seat the source faces this arrangement promotes by THE SEATING LAW (@ref
/// tf::intersect::graph::seat_named_planes), against every plane `world`
/// holds. Per face, in `descriptors` order: `plane_of`, the plane it is seated
/// on, and `orientations`, its winding in that plane's frame. Per plane the
/// faces found past `world.n_planes()`: `new_frames` — its smallest face's
/// own, from `face_frames` — and `new_names`. Returns the new plane count.
template <typename Index, typename Int, typename World, typename Descriptors,
          typename FaceFrames, typename ApplyToForm, typename GetPoint>
auto seat_plane_promoted_faces(
    const World &world, const Descriptors &descriptors,
    const FaceFrames &face_frames, const ApplyToForm &apply_to_form,
    const GetPoint &get_point, tf::buffer<Index> &plane_of,
    tf::buffer<tf::exact::plane_frame<Int>> &new_frames,
    tf::buffer<tf::exact::canonical_plane<Int>> &new_names,
    tf::buffer<std::int8_t> &orientations) -> Index {
  const auto n_world_planes = world.n_planes();
  tf::buffer<tf::exact::canonical_plane<Int>> names;
  names.allocate(descriptors.size());
  tf::parallel_for_each(
      tf::make_sequence_range(descriptors.size()),
      [&](std::size_t at) {
        const auto &descriptor = descriptors[at];
        names[at] = tf::intersect::graph::name_plane_face<Index, Int>(
            Index(descriptor.tag), descriptor.object, apply_to_form,
            get_point);
      },
      tf::checked(64));
  const auto standing_names = tf::make_mapped_range(
      tf::make_sequence_range(n_world_planes),
      [&world](Index plane) { return world.plane_name(plane); });
  tf::buffer<Index> new_plane_first;
  const auto n_new = tf::intersect::graph::seat_named_planes<Index, Int>(
      names, standing_names, n_world_planes, plane_of, new_plane_first);
  new_frames.allocate(std::size_t(n_new));
  new_names.allocate(std::size_t(n_new));
  for (Index r = 0; r < n_new; ++r) {
    const auto first = std::size_t(new_plane_first[std::size_t(r)]);
    new_frames[std::size_t(r)] = face_frames[first];
    new_names[std::size_t(r)] = names[first];
  }
  orientations.allocate(descriptors.size());
  tf::parallel_for_each(
      tf::make_sequence_range(descriptors.size()),
      [&](std::size_t at) {
        const auto plane = plane_of[at];
        const auto &carrier =
            plane < n_world_planes
                ? world.frame(plane)
                : new_frames[std::size_t(plane - n_world_planes)];
        const auto &descriptor = descriptors[at];
        apply_to_form(Index(descriptor.tag), [&](const auto &form) {
          orientations[at] = tf::exact::plane_frame_winding(
              carrier, form.faces()[descriptor.object], [&](auto corner) {
                return get_point(std::int16_t(descriptor.tag), Index(corner));
              });
        });
      },
      tf::checked);
  return n_new;
}

} // namespace tf::arrangement
