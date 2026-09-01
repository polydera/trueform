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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../exact/snap_parameter_to_edge.hpp"
#include "../../exact/vertex.hpp"
#include "./has_adjacent_split.hpp"
#include "./plane_refinement_physical_split.hpp"
#include "./select_split_proposals.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// Snap and select one sizing ring's physical split proposals.
///
/// Proposals are normalized on entry and compacted to the accepted rows for
/// the caller's accounting. The shared ring kernel states the next edge
/// frontier before materialization, so a rejected row does not erase sizing
/// continuity.
template <typename Index, typename Int, typename GetMeshPoint>
auto materialize_plane_refinement_ring_splits(
    tf::buffer<plane_refinement_physical_split<Index, Int>> &proposals,
    const GetMeshPoint &get_mesh_point,
    tf::buffer<plane_refinement_physical_split<Index, Int>> &physical)
    -> std::size_t {
  if (proposals.size() == 0)
    return 0;

  tf::parallel_for_each(
      proposals,
      [&get_mesh_point](plane_refinement_physical_split<Index, Int> &proposal) {
        const auto a = get_mesh_point(int(proposal.tag), proposal.u);
        const auto b = get_mesh_point(int(proposal.tag), proposal.v);
        proposal.parameter = tf::exact::snap_parameter_to_edge<Int>(
            proposal.parameter, tf::exact::pt3<Int>{a[0], a[1], a[2]},
            tf::exact::pt3<Int>{b[0], b[1], b[2]});
      },
      tf::checked);

  tf::buffer<std::size_t> selected;
  tf::arrangement::select_split_proposals<Int>(
      proposals,
      [](const plane_refinement_physical_split<Index, Int> &split) {
        return tf::arrangement::plane_refinement_physical_edge(split);
      },
      [](const plane_refinement_physical_split<Index, Int> &split) {
        return split.parameter;
      },
      [&physical](const plane_refinement_physical_split<Index, Int> &split) {
        return tf::arrangement::has_adjacent_split(
            tf::arrangement::plane_refinement_physical_edge(split),
            split.parameter, physical,
            [](const plane_refinement_physical_split<Index, Int> &accepted) {
              return tf::arrangement::plane_refinement_physical_edge(accepted);
            },
            [](const plane_refinement_physical_split<Index, Int> &accepted) {
              return accepted.parameter;
            });
      },
      selected);

  auto *write = proposals.begin();
  for (const auto index : selected)
    *write++ = proposals[index];
  proposals.erase_till_end(write);
  if (proposals.size() == 0)
    return 0;

  const auto base = physical.size();
  physical.reallocate(base + proposals.size());
  std::copy(proposals.begin(), proposals.end(), physical.begin() + base);
  std::inplace_merge(physical.begin(), physical.begin() + base, physical.end());
  return proposals.size();
}

} // namespace tf::arrangement
