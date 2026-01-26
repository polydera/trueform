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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership_like.hpp"

namespace tf::topology {
template <typename Range, typename Policy, typename Index>
auto compute_face_link_per_edge(const Range &faces,
                                const tf::face_membership_like<Policy> &blink,
                                tf::buffer<Index> &offsets,
                                tf::buffer<Index> &data) {
  if (!faces.size())
    return;
  auto result = std::tie(offsets, data);
  if constexpr (tf::static_size_v<decltype(faces[0])> != tf::dynamic_size)
    offsets.reserve(faces.size() * tf::static_size_v<decltype(faces[0])> + 1);
  else
    offsets.reserve(faces.size() * 3 + 1);

  auto aggregate_f =
      [](const std::pair<tf::buffer<Index>, tf::buffer<Index>> &local_result,
         std::tuple<tf::buffer<Index> &, tf::buffer<Index> &> &result) {
        auto &[l_offsets, l_data] = local_result;
        auto &[offsets, data] = result;
        auto old_data_size = data.size();
        data.reallocate(old_data_size + l_data.size());
        std::copy(l_data.begin(), l_data.end(), data.begin() + old_data_size);
        auto old_offsets_size = offsets.size();
        offsets.reallocate(old_offsets_size + l_offsets.size());
        auto it = offsets.begin() + old_offsets_size;
        for (auto l_offset : l_offsets) {
          *it++ = l_offset + old_data_size;
        }
      };

  auto task_f = [&](auto &&r,
                    std::pair<tf::buffer<Index>, tf::buffer<Index>> &pair) {
    auto &[offsets, ids] = pair;
    for (const auto &[face_id, face] : r) {
      Index size = face.size();
      for (Index next = 1; next < size; ++next) {
        offsets.push_back(ids.size());
        tf::face_edge_neighbors(blink, faces, Index(face_id),
                                Index(face[next - 1]), Index(face[next]),
                                std::back_inserter(ids));
      }
      offsets.push_back(ids.size());
      tf::face_edge_neighbors(blink, faces, Index(face_id),
                              Index(face[size - 1]), Index(face[0]),
                              std::back_inserter(ids));
    }
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::enumerate(faces), result,
      std::pair<tf::buffer<Index>, tf::buffer<Index>>{}, task_f, aggregate_f);
  offsets.push_back(data.size());
}

template <typename Range, typename Policy, typename Index>
auto compute_face_link_per_edge(const Range &faces,
                                const tf::face_membership_like<Policy> &blink,
                                tf::offset_block_buffer<Index, Index> &buff) {
  compute_face_link_per_edge(faces, blink, buff.offsets_buffer(),
                             buff.data_buffer());
}

} // namespace tf::topology
