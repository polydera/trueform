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
#include "../../core/checked.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/policy/buffer.hpp"
#include "../../core/policy/none.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../spatial/make_buffer_for_form.hpp"
#include "../classify/face_plane_info.hpp"
#include "./face_images.hpp"
#include "./placed_face_box.hpp"

#include <cstddef>

namespace tf::intersect {

/// The images of every face of `form`, read through the lattice exactly
/// as a leaf gather reads them: the corner offsets by one prefix over the
/// face sizes, then each face's corners, box and plane written into its
/// own rows.
template <typename Index, typename Int, typename Form, typename Lattice>
auto make_face_images(const Form &form, int tag, const Lattice &lattice)
    -> face_images<Index, Int> {
  face_images<Index, Int> images;
  const auto n_faces = std::size_t(form.size());
  auto &offsets = images.corners.offsets_buffer();
  offsets.allocate(n_faces + 1);
  offsets[0] = 0;
  for (std::size_t f = 0; f < n_faces; ++f)
    offsets[f + 1] = offsets[f] + Index(form.faces()[f].size());
  images.corners.data_buffer().allocate(std::size_t(offsets[n_faces]));
  images.boxes.allocate(n_faces);
  images.planes.allocate(n_faces);
  tf::parallel_for_each(
      tf::make_sequence_range(n_faces),
      [&](std::size_t f, auto &buff) {
        auto poly =
            tf::transformed(form[f] | tf::tag(buff), tf::frame_of(form));
        auto *corners = images.corners.data_buffer().begin() + offsets[f];
        const std::size_t m = poly.size();
        for (std::size_t k = 0; k < m; ++k)
          corners[k] = {lattice.flat_vertex(tag, Index(poly.indices()[k])),
                        lattice.point(tag, Index(poly.indices()[k]), poly[k])};
        const auto face = images.face(f);
        images.boxes[f] = placed_face_box(face);
        images.planes[f] = tf::exact::make_face_plane(face);
      },
      tf::spatial::make_buffer_for_form(form), tf::checked);
  return images;
}

} // namespace tf::intersect
