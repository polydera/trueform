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

#include "../core/point.hpp"
#include <cstdint>

namespace tf::exact {

using pt2 = tf::point<int32_t, 2>;
using pt3 = tf::point<int32_t, 3>;

/// Vertex with unique ID for SoS (Simulation of Simplicity) perturbation.
/// The ID determines the perturbation priority in exact predicates,
/// guaranteeing no degenerate (zero) results.
struct vertex {
  int id;
  pt3 pt;
};

} // namespace tf::exact
