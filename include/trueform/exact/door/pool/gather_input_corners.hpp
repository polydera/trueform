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

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// The one fact the placement tables do not carry: each face's corners in
/// their flat vertex numbering. A placement reads its incident faces and
/// never their corners, so the tables state the incidence and this states
/// its transpose. Everything else the naming pass reads is the tables' own.
///
/// The pass declines a face that is not a triangle. A name is an exact plane
/// through three original vertices, and a face of four corners standing on
/// one rounded plane need not stand on any exact one, so an arity this tier
/// cannot name refuses the whole scene and the door runs without it.
template <typename Index, typename ApplyToForm>
auto gather_input_corners(const ApplyToForm &apply_to_form, Index n_tags,
                          const tf::buffer<Index> &vertex_offsets,
                          const tf::buffer<Index> &face_offsets,
                          tf::blocked_buffer<Index, 3> &corners) -> bool {
  corners.allocate(std::size_t(face_offsets[std::size_t(n_tags)]));

  bool triangles = true;
  for (Index tag = 0; tag < n_tags; ++tag)
    apply_to_form(tag, [&, tag](const auto &form) {
      const auto vertex_base = vertex_offsets[std::size_t(tag)];
      const auto face_base = std::size_t(face_offsets[std::size_t(tag)]);
      const auto faces = form.faces();
      auto &table = corners;
      tf::parallel_for_each(
          tf::make_sequence_range(faces.size()),
          [&table, &triangles, faces, face_base,
           vertex_base](std::size_t id) {
            const auto face = faces[Index(id)];
            if (face.size() != 3) {
              triangles = false;
              return;
            }
            auto block = table[face_base + id];
            block[0] = vertex_base + Index(face[0]);
            block[1] = vertex_base + Index(face[1]);
            block[2] = vertex_base + Index(face[2]);
          },
          tf::checked);
    });
  return triangles;
}

} // namespace tf::exact::door::pool
