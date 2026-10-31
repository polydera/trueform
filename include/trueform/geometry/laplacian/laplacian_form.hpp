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
#include "../../linalg/symmetric_lower.hpp"

namespace tf::geometry {

/// The free rows of the form `k_s L + k_b L M^-1 L + tau M` a
/// @ref tf::geometry::laplacian_system states, as explicit matrices.
///
/// `free` is the block over the free columns, by free position: row `i` is
/// the free row `free_ids[i]`, and its columns are the free positions `j <= i`.
/// The block over the fixed columns is by point id: row `i` holds
/// `fixed_columns`, ascending, at `fixed_offsets`. One walk over each free
/// row states both blocks, so the factor and the right-hand side read the
/// same operator.
template <typename Index> struct laplacian_form {
  tf::symmetric_lower<Index> free;
  tf::buffer<Index> fixed_offsets;
  tf::buffer<Index> fixed_columns;
  tf::buffer<double> fixed_values;
};

} // namespace tf::geometry
