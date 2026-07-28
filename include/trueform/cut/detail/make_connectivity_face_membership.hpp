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

#include "../../core/algorithm/compact_ids_and_make_map.hpp"
#include "../../core/buffer.hpp"
#include "../../topology/face_membership.hpp"

#include <cstddef>
#include <utility>

namespace tf::cut::detail {

/// The temporary old-to-new map allocation becomes the membership-offset
/// allocation after the compaction barrier.
template <typename Index, typename Faces>
auto make_connectivity_face_membership(const Faces &faces,
                                       tf::buffer<Index> &flat_ids,
                                       Index bounded_id_count)
    -> tf::face_membership<Index> {
  tf::face_membership<Index> membership;
  const auto occurrence_count = flat_ids.size();
  const auto domain_size = static_cast<std::size_t>(bounded_id_count);
  if (occurrence_count && domain_size / occurrence_count >= 4) {
    auto [compact_id_count, old_to_new] = tf::compact_ids_and_make_map(
        tf::make_range(flat_ids), bounded_id_count);
    bounded_id_count = compact_id_count;
    // The sparse threshold leaves enough capacity for all compact offsets.
    membership.offsets_buffer() = std::move(old_to_new);
  }
  membership.build(faces, static_cast<std::size_t>(bounded_id_count),
                   occurrence_count);
  return membership;
}

} // namespace tf::cut::detail
