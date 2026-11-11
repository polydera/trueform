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
#include "../../../../core/blocked_buffer.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/faces.hpp"
#include "../../../../core/none.hpp"
#include "../../../../core/views/indirect_range.hpp"
#include "../../../face_membership.hpp"
#include "../../../manifold_edge_link.hpp"
#include "../clear_tet_written_rows.hpp"
#include "../compact_tet_dead_rows.hpp"
#include "../tet_execution_tuning.hpp"
#include "../tet_owner_complex.hpp"
#include "../tet_protected_count.hpp"
#include "../tet_site_position.hpp"
#include "./check_tet_segment_fixpoint.hpp"
#include "./measure_tet_phase_damage.hpp"
#include "./recover_tet_segments.hpp"
#include "./restore_tet_delaunay.hpp"
#include "../../../../core/algorithm/parallel_copy.hpp"
#include "../../../../core/algorithm/parallel_iota.hpp"
#include "./regions/tet_patch_tables.hpp"
#include "./state_tet_facet_segments.hpp"
#include "./tet_kept_rows.hpp"
#include "./tet_refusing_cut.hpp"
#include "./tet_segment_phase.hpp"
#include "./tet_segment_status.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Phase S: recover the protected `segments` and every edge bounding a face
/// of `faces`, read through `link`, the faces' triangles' manifold edge link
/// in block order, built here when none is given
/// (@ref state_tet_facet_segments), to fixpoint in the
/// unconstrained complex `owner`, whose lane is empty, leaving its rows as
/// they stand: the publish is the caller's barrier
/// (@ref compact_tetrahedralization). A
/// piece is kept once it is an edge of the complex, which under the
/// kernel's one tie rule is its strong Delaunay certificate, so the phase
/// reads no certificate (@ref recover_tet_segments), splitting each segment on
/// its frame `depth` bits deep, deeper where `phase.segment.deepens`, within
/// `phase.segment.site_budget` sites, handing every crossing its walk meets to
/// `cut` (@ref recover_tet_segments); a recovered segment's pieces are its
/// chain's consecutive sites. The rows the phase writes stay recorded for
/// the one restoration that follows the region labeling, except where the
/// phase damaged the complex more densely than
/// `tet_execution_tuning::phase_restore_density`
/// (@ref measure_tet_phase_damage): there the phase restores its own rows
/// before phase F reads them (@ref restore_tet_delaunay) and records them no
/// further, so that restoration's cascade and phase F's own work both run on
/// a Delaunay complex. `phase.segment.restored_before_faces` states which
/// ran. Returns false only when an append refused.
///
/// The phase is a diff on the tables `phase` holds: the faces of `faces`
/// that `due_faces` names are the ones due, their rims stated beside the
/// standing ones as blocks `edge_block` tickets each face to, and `link`
/// the link of their triangles alone, in the order due; only the segments
/// that statement adds and the rows a standing segment lost are recovered;
/// everything recovered before stays as it stands, its rows still recorded
/// for the one restoration. A phase holding no rim yet is the whole surface
/// on an unconstrained complex whose lane is empty.
template <bool Parallel, typename Owner, typename Work, typename Segments,
          typename Faces, typename Due, typename Link = tf::none_t,
          typename Cut = tet_refusing_cut>
auto recover_tet_segment_phase(
    Owner &owner, Work &work, const Segments &segments, const Faces &faces,
    const Due &due_faces, int depth,
    tet_segment_phase<typename Owner::index_type, typename Owner::coord_type>
        &phase,
    const Link &link = Link{}, const Cut &cut = Cut{}) -> bool {
  using Index = typename Owner::index_type;
  const bool whole = phase.edges.size() == 0;
  if constexpr (tf::core::verification)
    tf::core::verify(!whole || !tet_protected_count(owner),
                     "the segment phase starts on an empty lane");
  const auto due = tf::make_indirect_range(due_faces, faces);
  const std::size_t standing = phase.segment.status.size();
  const std::size_t stated = phase.edges.size();
  const auto live = [&phase](Index s) {
    return phase.segment.status[std::size_t(s)] != tet_segment_status::retired;
  };
  // THE REGIONS the round was asked for, stated on the link the phase holds:
  // the row table IS the mark every admission reads, and nothing else is
  // stated. O(triangles + facets + sites + rows), and nothing unasked.
  // THE STATEMENT OBEYS THE ROUND'S OWN LAW: the partition and the mark are
  // stated at the first opening whose surface the grain divides at all — the
  // answer read off the triangle count before the tables are paid for — and
  // CARRIED from there — a born
  // row is marked by its maker, a compaction carries the lane with the rows,
  // and a round covers whatever the serial path appended between them. A call
  // no phase of which meets the threshold states nothing and is the serial
  // kernel, paying nothing.
  const auto state_regions = [&](const auto &use,
                                 std::size_t triangles) -> std::size_t {
    auto &patches = phase.segment.site_patches;
    const std::size_t sites = tet_site_count(owner);
    if (!patches.size()) {
      if (!tet_execution_tuning::region_triangles ||
          triangles < 2 * tet_execution_tuning::region_triangles)
        return 0;
      state_tet_patch_tables(owner, due, use, phase.region_tables,
                             tet_execution_tuning::region_triangles);
      if (phase.region_tables.regions < 2)
        return 0;
      patches.allocate(phase.region_tables.site.size());
      tf::parallel_copy(phase.region_tables.site, patches);
      phase.segment.n_regions = std::size_t(phase.region_tables.regions);
    }
    // a site the frontier minted between rounds is nobody's, so every task
    // that reaches a row standing on it stops and the frontier does its work
    const std::size_t known = patches.size();
    if (known < sites) {
      patches.reallocate(sites);
      for (std::size_t site = known; site < sites; ++site)
        patches[site] = Index(-1);
    }
    if constexpr (tf::core::verification)
      tf::core::verify(patches.size() == sites,
                       "the patches name every site the complex holds");
    return phase.segment.n_regions;
  };
  tf::blocked_buffer<Index, 3> triangles;
  tf::face_membership<Index> membership;
  tf::manifold_edge_link<Index, 3> built;
  std::size_t n_triangles = 0;
  if constexpr (std::is_same_v<Link, tf::none_t>) {
    for (std::size_t f = 0; f < due.size(); ++f)
      n_triangles += due[f].size();
    triangles.allocate(n_triangles);
    std::size_t t = 0;
    for (std::size_t f = 0; f < due.size(); ++f)
      for (const auto &triangle : due[f]) {
        for (std::size_t k = 0; k < 3; ++k)
          triangles[t][k] = Index(triangle[k]);
        ++t;
      }
    membership.build(tf::make_faces(triangles), tet_site_count(owner));
    built.build(tf::make_faces(triangles), membership);
    state_tet_facet_segments(segments, due, built, live, tet_site_count(owner),
                             phase.segments, phase.segment_sites, phase.edges,
                             phase.holders, phase.facet_segments);
  } else {
    state_tet_facet_segments(segments, due, link, live, tet_site_count(owner),
                             phase.segments, phase.segment_sites, phase.edges,
                             phase.holders, phase.facet_segments);
    for (std::size_t f = 0; f < due.size(); ++f)
      n_triangles += due[f].size();
  }
  // THE PARTITION IS THE FIRST ENGAGED ROUND'S TO ASK FOR: a mesh no round
  // of which meets the threshold never asks, and states nothing.
  const auto partition = [&]() -> std::size_t {
    if constexpr (std::is_same_v<Link, tf::none_t>)
      return state_regions(built, n_triangles);
    else
      return state_regions(link, n_triangles);
  };
  phase.edge_block.reallocate(faces.size());
  for (std::size_t i = 0; i < due_faces.size(); ++i)
    phase.edge_block[std::size_t(due_faces[i])] = Index(stated + i);
  compact_tet_dead_rows<true>(owner);
  if (whole)
    clear_tet_written_rows(owner);
  owner._record_written = true;
  tet_owner_complex<Parallel, Owner, Work> complex(owner, work);
  const bool appended =
      recover_tet_segments(complex, phase.segments, whole ? 0 : standing,
                           depth, phase.segment, cut, partition);
  const bool restores =
      appended && measure_tet_phase_damage(owner) >
                      tet_execution_tuning::phase_restore_density;
  phase.segment.restored_before_faces =
      (!whole && phase.segment.restored_before_faces) || restores;
  if (restores) {
    phase.segment.unrestored +=
        restore_tet_delaunay(owner, phase.segment, tet_kept_rows<Index>{});
    clear_tet_written_rows(owner);
  }
  if constexpr (tf::core::verification)
    tf::core::verify(!appended || ((!whole || !tet_protected_count(owner)) &&
                                   check_tet_segment_fixpoint(owner, phase)),
                     "every recovered segment stands as a chain of edges");
  return appended;
}

} // namespace tf::topology::cdt::dt3
