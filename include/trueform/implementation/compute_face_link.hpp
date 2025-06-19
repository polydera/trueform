/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../face_edge_neighbors.hpp"
#include "../face_membership.hpp"
#include "../generate_offset_blocks.hpp"
#include "../offset_block_buffer.hpp"
#include "../sequence_range.hpp"

namespace tf::implementation {
template <typename Range, typename Index>
auto compute_face_link(const Range &faces,
                       const tf::face_membership<Index> &blink,
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

template <typename Range, typename Index>
auto compute_face_link(const Range &faces,
                       const tf::face_membership<Index> &blink,
                       tf::offset_block_buffer<Index, Index> &buff) {
  compute_face_link(faces, blink, buff.offsets_buffer(), buff.data_buffer());
}

} // namespace tf::implementation
