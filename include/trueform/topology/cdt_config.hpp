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

#include "./cdt_region_mode.hpp"

namespace tf {

/// @ingroup topology
/// @brief Configuration for a `make_cdt` build: the constraint-crossing
///        policy plus the region fact the labels read carries.
///
/// Implicitly constructible from `bool` (split_constraints) or
/// @ref tf::cdt_region_mode alone, so a call site may spell only the
/// part it cares about. `regions` selects what a
/// @ref tf::return_region_labels read states; the default entries always
/// filter by nesting parity.
struct cdt_config {
  bool split_constraints = true;
  cdt_region_mode regions = cdt_region_mode::nesting;

  constexpr cdt_config() = default;
  constexpr cdt_config(bool split) : split_constraints(split) {}
  constexpr cdt_config(cdt_region_mode r) : regions(r) {}
  constexpr cdt_config(bool split, cdt_region_mode r)
      : split_constraints(split), regions(r) {}
};

} // namespace tf
