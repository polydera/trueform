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
#include "../../../face_membership.hpp"
#include "../../../manifold_edge_link.hpp"
#include "../check_tetrahedralization_validity.hpp"
#include "../compact_tet_dead_rows.hpp"
#include "../compact_tetrahedralization.hpp"
#include "../tet_owner_complex.hpp"
#include "../tet_site_position.hpp"
#include "./check_tet_segment_fixpoint.hpp"
#include "./recover_tet_segments.hpp"
#include "./restore_tet_delaunay.hpp"
#include "./state_tet_facet_segments.hpp"
#include "./state_tet_segment_pieces.hpp"
#include "./tet_segment_phase.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Phase S: recover the protected `segments` and every edge bounding a face
/// of `faces`, read through `link`, the faces' triangles' manifold edge link
/// in block order, built here when none is given
/// (@ref state_tet_facet_segments), to fixpoint in the
/// unconstrained complex `owner`, whose lane is empty, and publish it. A
/// piece is kept once it is an edge of the complex, which under the
/// kernel's one tie rule is its strong Delaunay certificate, so the phase
/// reads no certificate (@ref recover_tet_segments), splitting each segment on
/// its frame `depth` bits deep, deeper where `phase.segment.deepens`, within
/// `phase.segment.site_budget` sites,
/// then restore the complex to Delaunay wherever flips reach, the pieces
/// barred (@ref restore_tet_delaunay), and state the pieces it recovered in
/// `phase.pieces`. Returns false only when an append refused, which
/// publishes no cells.
template <bool Parallel, typename Owner, typename Work, typename Segments,
          typename Faces, typename Link = tf::none_t>
auto recover_tet_segment_phase(
    Owner &owner, Work &work, const Segments &segments, const Faces &faces,
    int depth,
    tet_segment_phase<typename Owner::index_type, typename Owner::coord_type>
        &phase,
    const Link &link = Link{}) -> bool {
  using Index = typename Owner::index_type;
  if constexpr (tf::core::verification)
    tf::core::verify(!owner._n_protected,
                     "the segment phase starts on an empty lane");
  if constexpr (std::is_same_v<Link, tf::none_t>) {
    tf::blocked_buffer<Index, 3> triangles;
    std::size_t n_triangles = 0;
    for (std::size_t f = 0; f < faces.size(); ++f)
      n_triangles += faces[f].size();
    triangles.allocate(n_triangles);
    std::size_t t = 0;
    for (std::size_t f = 0; f < faces.size(); ++f)
      for (const auto &triangle : faces[f]) {
        for (std::size_t k = 0; k < 3; ++k)
          triangles[t][k] = Index(triangle[k]);
        ++t;
      }
    tf::face_membership<Index> membership;
    membership.build(tf::make_faces(triangles), tet_site_count(owner));
    tf::manifold_edge_link<Index, 3> built;
    built.build(tf::make_faces(triangles), membership);
    state_tet_facet_segments(segments, faces, built, phase.segments,
                             phase.edges, phase.holders);
  } else {
    state_tet_facet_segments(segments, faces, link, phase.segments, phase.edges,
                             phase.holders);
  }
  compact_tet_dead_rows<true>(owner);
  owner._written.clear();
  owner._record_written = true;
  tet_owner_complex<Parallel, Owner, Work> complex(owner, work);
  const bool appended =
      recover_tet_segments(complex, phase.segments, depth, phase.segment);
  phase.segment.unrestored +=
      appended ? restore_tet_delaunay(owner, phase.segment) : 0;
  owner._record_written = false;
  owner._written.clear();
  compact_tetrahedralization<true, true>(owner);
  if constexpr (tf::core::verification)
    tf::core::verify(!appended || check_tetrahedralization_validity(owner),
                     "the segment phase publishes a valid tetrahedralization");
  state_tet_segment_pieces(phase.segment, phase.pieces);
  if constexpr (tf::core::verification)
    tf::core::verify(!appended || check_tet_segment_fixpoint(owner, phase),
                     "every recovered segment stands as a chain of edges");
  return appended;
}

} // namespace tf::topology::cdt::dt3
