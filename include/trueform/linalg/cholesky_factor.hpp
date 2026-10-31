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

namespace tf {

/// @ingroup linalg
/// @brief The lower factor `L` of `P A P^T = L L^T`, one dense block per
/// supernode where @ref tf::cholesky_pattern::value_offsets places it; the
/// pattern states which row and column each value is.
struct cholesky_factor {
  tf::buffer<double> values;
};

} // namespace tf
