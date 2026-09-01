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
#include "../../core/offset_block_buffer.hpp"
#include "./make_plane_radial_fans.hpp"

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Implicit N-form arrangement descriptor: domains, bundles,
///        per-component tag, per-bundle tag-set, NM-edge wedges.
///
/// `domain_of_side[2c + s]` is the domain id on side `s` of component
/// `c`. Open components are self-merged so `d0 == d1`; intersecting
/// components are unified via NM-edge wedge merges. Nesting (inside /
/// outside an isolated shell) is *not* resolved here.
///
/// `bundle_of_component[c]` is the 3D-connected bundle id containing
/// `c`. `tag_of_component[c]` is the form tag of `c`.
/// `bundle_to_tags[b]` is the ascending-sorted unique list of form
/// tags whose components belong to bundle `b`.
///
/// `fans` are the arrangement's radial fans — one per fan piece, its
/// carrier planes in radial order, each page holding the live
/// occurrences that sit on it; `valid` is the wedge admission.
///
/// Build via @ref tf::csg::graph::make_arrangement_descriptor.
template <typename Index> struct arrangement_descriptor {
  tf::buffer<Index> domain_of_side;
  Index n_domains = 0;

  tf::buffer<Index> bundle_of_component;
  Index n_bundles = 0;

  tf::buffer<Index> tag_of_component;
  tf::offset_block_buffer<Index, Index> bundle_to_tags;

  tf::csg::graph::plane_radial_fans<Index> fans;
  /// Per fan: two or more pages stand around the piece, so a cyclic ring
  /// exists to pair. A lone page pairs with itself and would merge a
  /// component's own two sides; it is surfaced here, counted in
  /// `n_invalid_fans`, never silently walked.
  tf::buffer<char> valid;
  Index n_invalid_fans = 0;
};

} // namespace tf::csg::graph
