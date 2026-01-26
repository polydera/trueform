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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership_like.hpp"

namespace tf::topology {
template <typename Range, typename Policy, typename Index>
auto compute_face_link(const Range &faces,
                       const tf::face_membership_like<Policy> &blink,
                       tf::buffer<Index> &offsets, tf::buffer<Index> &data) {

  auto fill_f = [&](Index face_id, tf::buffer<Index> &ids) {
    const auto &face = faces[face_id];
    Index size = face.size();
    Index current = size - 1;
    for (Index next = 0; next < size; current = next++) {
      tf::face_edge_neighbors(blink, faces, face_id, Index(face[current]),
                              Index(face[next]), std::back_inserter(ids));
    }
  };
  tf::generate_offset_blocks(tf::make_sequence_range(faces.size()), offsets,
                             data, fill_f);
}

template <typename Range, typename Policy, typename Index>
auto compute_face_link(const Range &faces,
                       const tf::face_membership_like<Policy> &blink,
                       tf::offset_block_buffer<Index, Index> &buff) {
  compute_face_link(faces, blink, buff.offsets_buffer(), buff.data_buffer());
}

} // namespace tf::topology
