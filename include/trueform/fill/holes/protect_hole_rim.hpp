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
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../topology/cdt/dt3/compact_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/recovery/recover_tet_segments.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_status.hpp"
#include "../../topology/tetrahedralization_refusal.hpp"
#include "../hole_fill_status.hpp"
#include "../hole_split.hpp"
#include "./hole_rim.hpp"
#include "./hole_split_position.hpp"
#include "./hole_table.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::fill {

/// Recover the rim's edges as segments of its own complex, split at most
/// @ref tf::hole_split_depth times on the original edge's dyadic frame,
/// publish the complex canonically as the segment phase does, then state
/// the expanded rim once from the chains: its positions, corners and edges,
/// its sites being the complex's (`position_of_site`, `site_of_position`),
/// while `rim.sites` keeps the cycle the complex was built from. A refusal
/// names the rim edge settled first, the lowest of a round. A chain through
/// any site its own edge did not mint pinches the rim. Where `split_rim`
/// forbids touching an original edge, no split is admitted and a missing rim
/// edge refuses the rim.
template <typename PointsPolicy, typename Index, typename Int, typename RealT>
auto protect_hole_rim(const tf::points<PointsPolicy> &points, bool split_rim,
                      tf::fill::hole_table_scratch<Index, Int, RealT> &scratch,
                      tf::fill::hole_rim<Index, Int, RealT> &rim,
                      Index &offending) -> tf::hole_fill_status {
  using status_t = tf::topology::cdt::dt3::tet_segment_status;
  if (!scratch.dt.build(tf::make_range(rim.sites)))
    return scratch.dt.refusal() ==
                   tf::tetrahedralization_refusal::index_capacity
               ? tf::hole_fill_status::refused_resource
               : tf::hole_fill_status::refused_table;
  const std::size_t n = rim.size();
  const auto &first = scratch.dt.index_map().f();
  scratch.segments.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    scratch.segments[k] = {first[k], first[k + 1 == n ? 0 : k + 1]};
  auto &recovery = scratch.recovery;
  if (!tf::topology::cdt::dt3::recover_tet_segments(
          scratch.dt, scratch.segments, split_rim ? tf::hole_split_depth : 0,
          recovery))
    return tf::hole_fill_status::refused_resource;
  tf::topology::cdt::dt3::compact_tetrahedralization<true, true>(
      scratch.dt.owner());

  const auto pinched = [&recovery](std::size_t k) {
    const auto chain = recovery.chains[k];
    for (std::size_t j = 1; j + 1 < chain.size(); ++j) {
      const auto slot = std::size_t(chain[j].site);
      if (chain[j].numerator < 0 || slot < recovery.first_minted ||
          std::size_t(recovery.minted[slot - recovery.first_minted]) != k)
        return true;
    }
    return false;
  };
  std::size_t refused = n;
  for (std::size_t k = 0; k < n; ++k)
    if ((recovery.status[k] != status_t::recovered || pinched(k)) &&
        (refused == n || recovery.rounds[k] < recovery.rounds[refused]))
      refused = k;
  if (refused < n) {
    offending = Index(refused);
    switch (recovery.status[refused]) {
    case status_t::recovered:
      return tf::hole_fill_status::refused_invalid_rim;
    case status_t::depth:
      return split_rim ? tf::hole_fill_status::refused_protection_depth
                       : tf::hole_fill_status::refused_rim_split;
    case status_t::capacity:
      return tf::hole_fill_status::refused_resource;
    default:
      return tf::hole_fill_status::refused_invalid_rim;
    }
  }

  scratch.site_of_position.clear();
  scratch.positions.clear();
  scratch.corners.clear();
  scratch.rim_edges.clear();
  for (std::size_t k = 0; k < n; ++k) {
    const auto chain = recovery.chains[k];
    const auto &edge = rim.edges[k];
    const auto parameter = [&edge](std::int64_t numerator) {
      return std::uint8_t(edge.t0 == 0 ? numerator
                                       : tf::hole_split_scale - numerator);
    };
    for (std::size_t j = 0; j + 1 < chain.size(); ++j) {
      scratch.site_of_position.push_back(chain[j].site);
      if (j) {
        scratch.positions.push_back(tf::fill::hole_split_position(
            points, edge.v0, edge.v1, int(parameter(chain[j].numerator))));
        scratch.corners.push_back(Index(-1));
      } else {
        scratch.positions.push_back(rim.positions[k]);
        scratch.corners.push_back(rim.corners[k]);
      }
      auto subedge = edge;
      subedge.t0 = parameter(chain[j].numerator);
      subedge.t1 = parameter(chain[j + 1].numerator);
      scratch.rim_edges.push_back(subedge);
    }
  }
  std::swap(rim.positions, scratch.positions);
  std::swap(rim.corners, scratch.corners);
  std::swap(rim.edges, scratch.rim_edges);
  scratch.position_of_site.allocate(scratch.dt.n_sites());
  for (std::size_t k = 0; k < rim.size(); ++k)
    scratch.position_of_site[std::size_t(scratch.site_of_position[k])] =
        Index(k);
  return tf::hole_fill_status::filled;
}

} // namespace tf::fill
