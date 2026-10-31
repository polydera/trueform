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
#include <cstddef>

namespace tf::linalg {

/// The workspace one run of supernodes factors on: the trailing block of the
/// front in hand, the place of each of its rows, and the stack of updates
/// the run's supernodes still owe their parents, packed by columns and
/// pushed in ascending order, so a supernode's children are the stack's top.
template <typename Index> struct cholesky_front_scratch {
  tf::buffer<double> tail;
  tf::buffer<Index> position;
  tf::buffer<Index> relative;
  tf::buffer<double> stack;
  tf::buffer<Index> pending;
  tf::buffer<std::size_t> pending_offsets;
};

} // namespace tf::linalg
