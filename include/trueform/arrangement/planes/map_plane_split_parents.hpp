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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_group_router.hpp"
#include "tbb/parallel_sort.h"

#include <cstddef>

namespace tf::arrangement {

/// Say a wave's splits in the local tier's own currency.
///
/// The port has just moved every immutable carrier the diff named, so the
/// router translates those statements to their current local parent. Local
/// statements already name that parent. Ordering the result by current parent
/// is the ascending, one-block-per-parent shape
/// @ref tf::arrangement::propose_plane_split_pieces reads.
///
/// False rejects the call: an immutable edge the router cannot answer, or two
/// statements that name one current parent. A split is a fact of that current
/// canonical group and may be stated only once in one wave.
template <typename Index>
auto map_plane_split_parents(const tf::buffer<Index> &split_edge,
                             const tf::buffer<Index> &split_tier,
                             const tf::buffer<Index> &split_offsets,
                             const tf::buffer<Index> &split_data,
                             const tf::buffer<Index> &group_router,
                             tf::buffer<Index> &parents,
                             tf::buffer<Index> &parent_offsets,
                             tf::buffer<Index> &parent_data) -> bool {
  struct parent_t {
    Index parent;
    Index begin;
    Index end;
  };
  parents.clear();
  parent_offsets.clear();
  parent_data.clear();
  parent_offsets.push_back(Index(0));
  if (split_edge.size() == 0)
    return split_tier.size() == 0 && split_offsets.size() <= 1 &&
           split_data.size() == 0;
  if (split_tier.size() != split_edge.size() ||
      split_offsets.size() != split_edge.size() + 1 ||
      split_offsets[0] != Index(0) ||
      split_offsets[split_edge.size()] != Index(split_data.size()))
    return false;

  const auto parent_of = [&split_edge, &split_tier,
                          &group_router](std::size_t at) {
    const auto edge = split_edge[at];
    if (edge < Index(0))
      return Index(-1);
    if (split_tier[at] == Index(0))
      return resolve_plane_group_router(group_router, edge, Index(-1));
    return split_tier[at] == Index(1) ? edge : Index(-1);
  };
  if (tf::parallel_contains(
          tf::make_sequence_range(split_edge.size()),
          [&parent_of](std::size_t at) { return parent_of(at) < Index(0); },
          tf::checked))
    return false;

  tf::buffer<parent_t> ordered;
  ordered.allocate(split_edge.size());
  tf::parallel_for_each(
      tf::make_sequence_range(split_edge.size()),
      [&](std::size_t at) {
        ordered[at] = {parent_of(at), split_offsets[at], split_offsets[at + 1]};
      },
      tf::checked);
  tbb::parallel_sort(ordered.begin(), ordered.end(),
                     [](const parent_t &x, const parent_t &y) {
                       return x.parent < y.parent;
                     });
  if (tf::parallel_contains(
          tf::make_sequence_range(ordered.size()),
          [&ordered](std::size_t at) {
            return at != 0 && ordered[at - 1].parent >= ordered[at].parent;
          },
          tf::checked)) {
    parent_offsets.clear();
    return false;
  }

  parents.allocate(ordered.size());
  parent_offsets.reallocate(ordered.size() + 1);
  parent_data.allocate(split_data.size());
  Index rows = 0;
  for (std::size_t at = 0; at < ordered.size(); ++at) {
    rows += ordered[at].end - ordered[at].begin;
    parent_offsets[at + 1] = rows;
  }
  tf::parallel_for_each(
      tf::make_sequence_range(ordered.size()),
      [&](std::size_t at) {
        parents[at] = ordered[at].parent;
        auto write = std::size_t(parent_offsets[at]);
        for (auto cut = ordered[at].begin; cut < ordered[at].end; ++cut)
          parent_data[write++] = split_data[std::size_t(cut)];
      },
      tf::checked);
  return true;
}

} // namespace tf::arrangement
