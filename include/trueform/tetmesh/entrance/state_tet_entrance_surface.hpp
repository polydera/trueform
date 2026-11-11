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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../topology/make_face_membership.hpp"
#include "../../topology/make_manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "./tet_entrance_surface.hpp"
#include "./tet_entrance_workspace.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// State `surface` from the `n_facets` facets the merge (@ref
/// merge_tet_facets) left in `work` — its triangles, their origins and
/// facets — and the counts `thrown` it threw out and added, `sources` naming
/// the input face each original triangle cuts and `link(t)` triangle `t`'s
/// manifold edge peers, read wherever the caller holds them: the triangles
/// ordered facet by facet, each with its parent, and into `stated` their
/// link in that order. The statement is a diff on the tables `surface`
/// holds: the facets are appended after the ones standing, numbered on from
/// them, their triangles after the standing triangles, and `stated` holds
/// the link of the appended triangles alone, a peer among the standing ones
/// read as a free edge, so a first statement over an empty surface states
/// it whole.
template <typename Index, typename RealT, typename Int, typename Sources,
          typename Link>
auto state_tet_entrance_surface(
    tet_entrance_surface<Index, RealT, Int> &surface, Index n_facets,
    const std::array<std::size_t, 3> &thrown, const Sources &sources,
    const Link &link, tet_entrance_workspace<Index, Int> &work,
    tf::manifold_edge_link<Index, 3> &stated) -> void {
  auto &triangles = work.triangles;
  auto &origin = work.origin;
  auto &facet_of = work.facet_of;
  surface.dropped_degenerate += thrown[0] + thrown[1];
  surface.stitched_triangles += thrown[2];
  auto &offsets = surface.facets;
  if (!offsets.size())
    offsets.push_back(Index(0));
  const std::size_t base = offsets.size() - 1;
  const std::size_t first = surface.triangles.size();
  // a counting sort whose keys are about as many as its records, a facet
  // being mostly one triangle: blocked counts would cost blocks x facets, so
  // the counts, their prefix and the cursor walk are the one serial pass
  offsets.reallocate(base + std::size_t(n_facets) + 1);
  tf::parallel_fill(tf::drop(offsets, base + 1), Index(0));
  for (auto facet : facet_of)
    ++offsets[base + std::size_t(facet) + 1];
  for (std::size_t f = base; f < base + std::size_t(n_facets); ++f)
    offsets[f + 1] += offsets[f];
  auto &slot = work.slot;
  slot.allocate(triangles.size());
  {
    auto &next = work.cursor;
    next.allocate(std::size_t(n_facets));
    tf::parallel_copy(tf::slice(offsets, base, base + std::size_t(n_facets)),
                      next);
    for (std::size_t t = 0; t < triangles.size(); ++t)
      slot[t] = Index(std::size_t(next[std::size_t(facet_of[t])]++) - first);
  }
  surface.triangles.reallocate(first + triangles.size());
  surface.parents.reallocate(first + triangles.size());
  tf::parallel_for_each(
      tf::make_sequence_range(triangles.size()),
      [&surface, &triangles, &origin, &sources, &slot, first](std::size_t t) {
        const auto at = first + std::size_t(slot[t]);
        for (std::size_t k = 0; k < 3; ++k)
          surface.triangles[at][k] = triangles[t][k];
        surface.parents[at] = Index(sources[std::size_t(origin[t])]);
      },
      tf::checked);
  const bool fresh_link = thrown[0] || thrown[1] || thrown[2];
  if (fresh_link) {
    const auto merged = tf::make_polygons(tf::drop(surface.triangles, first),
                                          surface.points.points());
    const auto membership = tf::make_face_membership<Index>(merged);
    stated = tf::make_manifold_edge_link<Index>(merged | tf::tag(membership));
  } else {
    stated.data_buffer().allocate(triangles.size() * 3);
    tf::parallel_for_each(
        tf::make_sequence_range(triangles.size()),
        [&stated, &link, &slot](std::size_t t) {
          const auto peers = link(t);
          auto placed = stated[std::size_t(slot[t])];
          for (std::size_t k = 0; k < 3; ++k) {
            const Index p = peers[k].face_peer;
            placed[k] = {p >= 0 ? slot[std::size_t(p)] : p};
          }
        },
        tf::checked);
  }
}

} // namespace tf::tetmesh
