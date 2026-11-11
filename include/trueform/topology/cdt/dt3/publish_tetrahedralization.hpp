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
#include "./compact_tetrahedralization.hpp"

namespace tf::topology::cdt::dt3 {

/// Publish the complex in canonical order: once after a mutation, nothing
/// while its rows already stand in that order.
template <bool Parallel, typename Owner>
auto publish_tetrahedralization(Owner &owner) -> void {
  if (!owner._canonical)
    compact_tetrahedralization<Parallel, true>(owner);
}

} // namespace tf::topology::cdt::dt3
