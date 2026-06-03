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
#include "./make_non_manifold_edge_fans.hpp"

namespace tf::cut {

/// @ingroup cut
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
/// `fans` + `reps` carry the canonical NM-edge wedge info.
///
/// Build via @ref tf::cut::make_arrangement_descriptor.
template <typename Index> struct arrangement_descriptor {
  tf::buffer<Index> domain_of_side;
  Index n_domains = 0;

  tf::buffer<Index> bundle_of_component;
  Index n_bundles = 0;

  tf::buffer<Index> tag_of_component;
  tf::offset_block_buffer<Index, Index> bundle_to_tags;

  tf::cut::non_manifold_edge_fans<Index> fans;
  tf::buffer<Index> reps;
};

} // namespace tf::cut
