/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/faces.hpp"
#include "../manifold_edge_link_like.hpp"
#include "./apply_face_flip_mask.hpp"
#include "./make_consistent_face_orientation_plan.hpp"

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Orient faces consistently using weighted voting.
///
/// Plans every reversal against the untouched winding, then applies the plan
/// once. The manifold-edge component is the carrier: an orientable component
/// comes back consistent, and one whose parity contradicts is left exactly as
/// it was.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The manifold edge link policy type.
/// @tparam Range The weights range type.
/// @param faces The faces range (modified in place).
/// @param link The manifold edge link built from exactly those face slots.
/// @param weights Per-face weights for voting (e.g., face areas).
/// @return `true` when every component is orientable and now consistent.
template <typename Policy, typename Policy1, typename Range>
auto orient_faces_consistently(tf::faces<Policy> &faces,
                               const tf::manifold_edge_link_like<Policy1> &link,
                               const Range &weights) -> bool {
  auto plan =
      tf::topology::make_consistent_face_orientation_plan(faces, link, weights);
  tf::topology::apply_face_flip_mask(faces, plan.flip_mask);
  return plan.orientable;
}
} // namespace tf::topology
