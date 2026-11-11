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
#include "../../topology/cdt/dt3/tet_facet_owner.hpp"

namespace tf::tetmesh {

/// An input face and a facet it lies in.
template <typename Index> struct tet_face_facet {
  Index face;
  Index facet;
};

/// The tables the index map's emitters walk on, constructed by the mesher
/// and handed in: the buckets' scratch, runs and counts
/// (@ref emit_tet_face_children), and the face-facet records
/// (@ref state_tet_face_facets).
template <typename Index> struct tet_face_map_workspace {
  tf::buffer<tf::topology::cdt::dt3::tet_facet_owner<Index>> scratch;
  tf::buffer<Index> runs;
  tf::buffer<Index> counts;
  tf::buffer<tet_face_facet<Index>> face_facets;
};

} // namespace tf::tetmesh
