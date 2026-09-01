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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./dedup_plane_loop.hpp"
#include "./face_descriptor.hpp"
#include "./loop.hpp"
#include "./vertex.hpp"
#include <cstddef>

namespace tf::intersect::graph {

/// The base loop of every face on the carrier, in block order.
///
/// A block is one face's claims — its pair records and its deliveries,
/// either of which may be empty — so the loop it extracts and the
/// descriptor that names it are one block position: the aggregation runs
/// in input-block order and the offsets it builds ARE the face carrier.
template <typename Index, typename Int, typename Records, typename Deliveries,
          typename ApplyToFace, typename GetPoint>
auto build_plane_loops(const Records &records, const Deliveries &deliveries,
                       const ApplyToFace &apply_to_face,
                       const GetPoint &get_point,
                       tf::buffer<face_descriptor<Index>> &descriptors,
                       tf::offset_block_buffer<Index, vertex<Index>> &loops)
    -> void {
  using vertex_t = vertex<Index>;
  const auto n = records.size();
  loops.offsets_buffer().allocate(n + 1);
  loops.offsets_buffer()[0] = 0;
  std::size_t loop_i = 1;

  struct local_t {
    tf::buffer<Index> sizes;
    tf::buffer<vertex_t> dirty;
    tf::buffer<vertex_t> data;
    tf::buffer<loop_node<Index, Int>> work;
    tf::buffer<face_descriptor<Index>> descs;
  };

  auto task = [&](auto &&range, local_t &local) {
    auto apply_to_face_f = apply_to_face;
    auto get_point_f = get_point;
    local.sizes.allocate(range.size());
    auto sit = local.sizes.begin();
    for (const auto block : range) {
      const auto old_size = local.data.size();
      auto &&recs = records[block];
      auto &&dels = deliveries[block];
      const Index tag =
          recs.size() != 0 ? Index(recs[0].tag) : Index(dels[0].tag);
      const auto object = recs.size() != 0 ? recs[0].object : dels[0].object;
      local.dirty.clear();
      apply_to_face_f(tag, object, [&](const auto &face) {
        extract_loop<Index, Int>(recs, dels, face, tag, get_point_f,
                                 local.work, local.dirty);
      });
      dedup_plane_loop<Index>(local.dirty, local.data);
      local.descs.push_back({tag, object});
      *sit++ = Index(local.data.size() - old_size);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.data, loops.data_buffer());
    tf::core::append(local.descs, descriptors);
    for (auto sz : local.sizes) {
      loops.offsets_buffer()[loop_i] = loops.offsets_buffer()[loop_i - 1] + sz;
      ++loop_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(tf::make_sequence_range(n), tf::none,
                                         local_t{}, task, agg);
}

} // namespace tf::intersect::graph
