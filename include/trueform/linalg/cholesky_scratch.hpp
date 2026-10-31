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
#include "../core/buffer.hpp"
#include "./cholesky/cholesky_front_scratch.hpp"
#include "./cholesky/minimum_degree_scratch.hpp"
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief The workspace a factorization and its solves walk on, owned by the
/// caller and reused across them.
///
/// @tparam Index The index type rows and entries are numbered in.
template <typename Index> struct cholesky_scratch {
  tf::linalg::minimum_degree_scratch<Index> ordering;
  tf::buffer<Index> minimum_degree;
  tf::buffer<Index> unordered_parent;
  tf::buffer<Index> post;
  tf::buffer<Index> children;
  tf::buffer<Index> siblings;
  tf::buffer<Index> stack;
  tf::buffer<Index> parent;
  tf::buffer<Index> counts;
  tf::buffer<Index> owner;
  tf::buffer<Index> fundamental;
  tf::buffer<Index> start;
  tf::buffer<Index> into;
  tf::buffer<double> zeros;
  tf::buffer<Index> stamp;
  tf::buffer<double> work;
  tf::buffer<double> lower_values;
  tf::buffer<std::size_t> handoff_offsets;
  tf::buffer<double> handoff;
  tf::buffer<char> refused;
  tf::linalg::cholesky_front_scratch<Index> front;
  tf::buffer<double> permuted;
  tf::buffer<double> gathered;
  tf::buffer<double> residual;
  tf::buffer<double> scale;
};

} // namespace tf
