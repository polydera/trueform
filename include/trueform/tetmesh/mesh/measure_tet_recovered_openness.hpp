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
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_facet_owner.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/has_holes.hpp"
#include "../entrance/count_tet_open_edges.hpp"
#include "./emit_tet_face_children.hpp"
#include "./gather_tet_face_owners.hpp"
#include "./tet_face_map_workspace.hpp"
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// What the recovered surface leaves open: its `edges`
/// (@ref tet_open_edges), and whether a rim of its free edges is a hole
/// (@ref tf::has_holes) rather than a flap.
struct tet_recovered_openness {
  tet_open_edges edges;
  bool holed = false;
};

/// Read what the recovered surface of `faces` leaves open in the complex
/// `owner`: the lane facets each face `status` says was recovered owns,
/// gathered into `records` as the complex stands (@ref
/// gather_tet_face_owners) and emitted over the sites
/// (@ref emit_tet_face_children) into `children`, their edges counted on
/// `steps` and their rims read through `membership`, the caller's own.
template <typename Index, typename Owner, typename Faces, typename Status>
auto measure_tet_recovered_openness(
    const Owner &owner, const Faces &faces, const Status &status,
    tf::buffer<tf::topology::cdt::dt3::tet_facet_owner<Index>> &records,
    tet_face_map_workspace<Index> &work,
    tf::buffer<tet_edge_step<Index>> &steps,
    tf::face_membership<Index> &membership,
    tf::offset_block_buffer<Index, std::array<Index, 3>> &children)
    -> tet_recovered_openness {
  gather_tet_face_owners(owner, records);
  emit_tet_face_children<Index>(
      owner, faces, status,
      tf::make_sequence_range(
          Index(tf::topology::cdt::dt3::tet_site_count(owner))),
      records, work, children);
  tet_recovered_openness openness;
  const auto &triangles = children.data_buffer();
  openness.edges = count_tet_open_edges(triangles, steps);
  if (openness.edges.boundary) {
    const auto recovered = tf::make_faces(tf::make_range(triangles));
    membership.build(recovered, tf::topology::cdt::dt3::tet_site_count(owner));
    openness.holed = tf::has_holes(recovered, membership);
  }
  return openness;
}

} // namespace tf::tetmesh
