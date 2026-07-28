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
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "./region_triangulation_types.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::cut::detail {

template <typename Index, typename Int, typename Local,
          typename MakeEmit>
auto promote_split_faces_applied(
    const tf::cut::detail::promoted_face_candidates<Index> &candidates,
    Index n_intersection_points, const MakeEmit &make_emit,
    Local &&initial,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        &triangles,
    tf::buffer<Index> &failed,
    tf::buffer<tf::intersect::graph::face_descriptor<Index>>
        &promoted,
    tf::buffer<Index> &promoted_interior_counts,
    Index &promoted_interior_base,
    tf::buffer<
        tf::cut::detail::region_triangulation_merge<Index>>
        &pending_merges) -> void {
  promoted_interior_counts.clear();
  if (candidates.faces.size() == 0)
    return;

  tf::buffer<tf::point<Int, 3>> interior_points;
  const Index interior_base = Index(extra_points.size());
  promoted_interior_base = interior_base;
  auto task = [&](auto &&range,
                  std::remove_reference_t<Local> &local) {
    auto emit = make_emit();
    for (auto &&[candidate_index, candidate] : range)
      emit(Index(candidate_index), candidate, local);
  };
  auto aggregate = [&](const std::remove_reference_t<Local> &local,
                       const tf::none_t &) {
    const Index base =
        interior_base + Index(interior_points.size());
    tf::core::append(local.interior_points, interior_points);
    const std::size_t first = triangles.size();
    tf::core::append(local.triangles, triangles);
    if (local.interior_points.size())
      for (std::size_t triangle = first;
           triangle < triangles.size(); ++triangle)
        for (int corner = 0; corner < 3; ++corner) {
          auto &vertex =
              triangles[triangle][std::size_t(corner)];
          if (vertex.source ==
                  tf::intersect::graph::vertex_source::created &&
              std::make_signed_t<Index>(vertex.id) < 0)
            vertex.id = n_intersection_points + base +
                        (-vertex.id - 1);
        }
    Index offset = Index(first);
    for (std::size_t candidate = 0;
         candidate < local.counts.size(); ++candidate) {
      if (!local.ok[candidate])
        failed.push_back(Index(ranges.size()));
      ranges.push_back(
          {offset, offset + local.counts[candidate]});
      offset += local.counts[candidate];
    }
    // A weld can retire a placeholder the same way it retires any vertex,
    // and a merge record carries the id in its keys as well as in its
    // identity — all three rebase, or the table names a vertex that does
    // not exist.
    const std::size_t first_merge = pending_merges.size();
    tf::core::append(local.workspace.merges, pending_merges);
    if (local.interior_points.size())
      for (std::size_t merge = first_merge;
           merge < pending_merges.size(); ++merge) {
        auto rebase = [&](Index &id) {
          if (std::make_signed_t<Index>(id) < 0)
            id = n_intersection_points + base + (-id - 1);
        };
        rebase(pending_merges[merge].from[1]);
        rebase(pending_merges[merge].to_key[1]);
        if (pending_merges[merge].to.source ==
            tf::intersect::graph::vertex_source::created)
          rebase(pending_merges[merge].to.id);
      }
    tf::core::append(local.interior_counts,
                     promoted_interior_counts);
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::zip(tf::make_sequence_range(candidates.faces.size()),
              tf::make_range(candidates.faces)),
      tf::none, std::forward<Local>(initial), task, aggregate);
  tf::core::append(interior_points, extra_points);
  for (const auto &candidate : candidates.faces)
    promoted.push_back({candidate[0], candidate[1]});
}

} // namespace tf::cut::detail
