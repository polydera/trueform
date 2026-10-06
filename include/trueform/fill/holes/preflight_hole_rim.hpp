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
#include "../../core/polygons.hpp"
#include "../../topology/boundary_rims.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_rim.hpp"
#include "./state_hole_rim_cycle.hpp"
#include "./validate_hole_rim.hpp"

namespace tf::fill {

/// Validate one rim and, where it passes, state its cycle for filling.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT, typename Converter>
auto preflight_hole_rim(const tf::polygons<Policy> &polygons,
                        const tf::face_membership_like<MembershipPolicy> &fm,
                        const tf::boundary_rims<Index> &rims, Index rim,
                        const Converter &converter,
                        tf::fill::hole_rim_scratch<Index, Int> &scratch,
                        tf::fill::hole_rim<Index, Int, RealT> &out)
    -> tf::fill::hole_preflight_verdict<Index> {
  const auto verdict = tf::fill::validate_hole_rim(polygons, fm, rims, rim,
                                                   converter, scratch);
  if (verdict.status != tf::hole_fill_status::filled)
    return verdict;
  tf::fill::state_hole_rim_cycle(polygons, rims, rim, scratch, out);
  return verdict;
}

} // namespace tf::fill
