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
#include "./flip_tet_facet.hpp"
#include "./remove_tet_edge_by_flips.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_flip_constraints.hpp"
#include "./tet_flip_scratch.hpp"
#include "./tet_orient3d_sites.hpp"
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Remove the facet a b c of `face` = [a, b, c, d] by flips: 2-3 where the two
/// apexes see each other through it, otherwise by removing the one of its
/// edges that is not locally convex (@ref remove_tet_edge_by_flips). A
/// protected facet is left alone. The facet is not a hull facet. Returns
/// whether it was removed.
template <typename Owner, typename IsSegment>
auto remove_tet_facet_by_flips(
    Owner &owner, tet_flip_scratch<typename Owner::index_type> &scratch,
    const tet_edge_ref<typename Owner::index_type> &face,
    const tet_flip_constraints<typename Owner::index_type,
                               typename Owner::coord_type,
                               typename Owner::int_type> &fc,
    const IsSegment &is_segment) -> bool {
  using Index = typename Owner::index_type;
  if (tet_facet_is_protected(owner, face.cell, tet_edge_oppo(face)))
    return false;
  const auto below = [&owner](Index w, Index x, Index y, Index z) {
    return tet_orient3d_below_sites(owner, w, x, y, z);
  };
  const auto c = tet_edge_corners(owner, face);
  const Index pa = c[0], pb = c[1], pc = c[2], pd = c[3];
  const Index pe = tet_edge_corners(owner, tet_edge_across(owner, face))[3];
  assert(pd != Owner::infinite && pe != Owner::infinite);
  tet_edge_ref<Index> flipedge = face;
  bool reducible = false;
  if (below(pa, pb, pd, pe) > 0) {
    if (below(pb, pc, pd, pe) > 0) {
      if (below(pc, pa, pd, pe) > 0)
        reducible = true;
      else
        flipedge = tet_edge_turn_back(face);
    } else
      flipedge = tet_edge_turn(face);
  }
  if (reducible) {
    const auto rows = flip_tet_facet(owner, face.cell, tet_edge_oppo(face));
    tet_ring_fresh(scratch, rows);
    return true;
  }
  return remove_tet_edge_by_flips(owner, scratch, flipedge, fc, is_segment) ==
         2;
}

} // namespace tf::topology::cdt::dt3
