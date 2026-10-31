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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/manifold_edge_peer.hpp"
#include <cstddef>

namespace tf::exact::door {

/// Per flat vertex whether it stands on a boundary edge of its form, an
/// edge one face holds: the rim. Only a rim vertex can leave a result
/// open, so only a rim vertex is a candidate for the band's landing.
template <typename Index, typename ApplyToForm>
auto mark_rim_vertices(const ApplyToForm &apply_to_form, Index n_tags,
                       const tf::buffer<Index> &vertex_offsets,
                       tf::buffer<char> &rim) -> void {
  rim.allocate(std::size_t(vertex_offsets[std::size_t(n_tags)]));
  tf::parallel_fill(rim, char(0));
  for (Index tag = 0; tag < n_tags; ++tag)
    apply_to_form(tag, [&rim, &vertex_offsets, tag](const auto &form) {
      const auto base = std::size_t(vertex_offsets[std::size_t(tag)]);
      const auto faces = form.faces();
      const auto &link = form.manifold_edge_link();
      tf::parallel_for_each(
          tf::make_sequence_range(Index(faces.size())),
          [&rim, &link, faces, base](Index f) {
            const auto corners = faces[f];
            const auto n = std::size_t(corners.size());
            for (std::size_t s = 0; s < n; ++s)
              if (link[f][s].face_peer ==
                  tf::manifold_edge_peer<Index>::boundary) {
                rim[base + std::size_t(corners[s])] = char(1);
                rim[base + std::size_t(corners[(s + 1) % n])] = char(1);
              }
          },
          tf::checked);
    });
}

} // namespace tf::exact::door
