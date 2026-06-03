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
#include "../../core/views/sequence_range.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Per-component side selection for a boolean result mesh.
///
/// For each component `c` in `[0, n_components)`, returns:
/// - `1`  if the result interior is on side 1 of the component (emit
///        faces with stored winding).
/// - `0`  if the result interior is on side 0 (emit faces with
///        reversed winding so output normals face outward from the
///        result).
/// - `-1` if both sides have the same membership in the result, i.e.
///        the component is fully inside or fully outside — skip its
///        faces (Lévy boundary rule: a face is on the result
///        boundary iff `E` differs across it).
///
/// @param desc Arrangement descriptor (provides `domain_of_side` +
///             `n_components` derived from `bundle_of_component`).
/// @param domain_membership Per-domain bool from
///        @ref tf::csg::graph::evaluate_per_domain.
template <typename Index>
auto compute_chosen_sides(
    const tf::cut::arrangement_descriptor<Index> &desc,
    const tf::buffer<bool> &domain_membership)
    -> tf::buffer<std::int8_t> {
  const Index n_components =
      static_cast<Index>(desc.bundle_of_component.size());
  tf::buffer<std::int8_t> chosen;
  chosen.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_for_each(tf::make_sequence_range(n_components), [&](Index c) {
    const bool s0 = domain_membership[desc.domain_of_side[2 * c + 0]];
    const bool s1 = domain_membership[desc.domain_of_side[2 * c + 1]];
    if (s0 == s1)
      chosen[c] = std::int8_t(-1);
    else
      chosen[c] = s1 ? std::int8_t(1) : std::int8_t(0);
  });
  return chosen;
}

} // namespace tf::csg::graph
