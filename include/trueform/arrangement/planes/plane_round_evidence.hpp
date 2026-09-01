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
#include "../../exact/meta.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_statement.hpp"

namespace tf::arrangement {

/// What one round of triangulation saw: the planes that refused, the
/// identities their resolve-mode rebuild could not tell apart, and the
/// statements and proposals it made — stated in one breath, so their
/// correspondence is position. The wave consumes exactly this.
template <typename Index, typename Int> struct plane_round_evidence {
  using param_t = typename tf::exact::meta<Int>::param_type;

  tf::buffer<Index> refused;
  tf::buffer<plane_flat_weld<Index>> welds;
  tf::buffer<plane_recovery_statement<Index, param_t>> statements;
  tf::buffer<plane_recovery_proposal<Index, param_t>> topology;
  plane_arrangement_census census;
};

} // namespace tf::arrangement
