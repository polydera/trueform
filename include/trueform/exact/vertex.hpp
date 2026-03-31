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

namespace tf::exact {

template <typename Int> using pt2 = tf::point<Int, 2>;

template <typename Int> using pt3 = tf::point<Int, 3>;

/// Vertex with unique ID for SoS (Simulation of Simplicity) perturbation.
/// The ID determines the perturbation priority in exact predicates,
/// guaranteeing no degenerate (zero) results.
template <typename Index, typename Int> struct vertex {
  Index id;
  pt3<Int> pt;
};

} // namespace tf::exact
