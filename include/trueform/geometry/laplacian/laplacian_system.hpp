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

namespace tf::geometry {

/// The discrete operators one triangle list states: a weighted Laplacian in
/// compressed rows whose columns ascend, and the barycentric lumped mass.
///
/// An edge of weight `w` is stored as `-w` off the diagonal, and the diagonal
/// holds the row's sum of weights, so `L` is positive semidefinite where
/// every weight is positive. The opposite convention, `+w` off the diagonal,
/// states the same biharmonic `L M^-1 L` and the same solution of `L x = 0`
/// on the free rows.
///
/// @tparam Index The index type the triangles are numbered in.
template <typename Index> struct laplacian_system {
  tf::buffer<Index> row_offsets;
  tf::buffer<Index> columns;
  tf::buffer<double> weights;
  tf::buffer<double> diagonal;
  tf::buffer<double> mass;

  auto size() const -> std::size_t { return mass.size(); }
};

} // namespace tf::geometry
