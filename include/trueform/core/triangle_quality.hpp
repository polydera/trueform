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
#include "./constants.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief The quality of a triangle from its doubled area and its longest side
/// squared: `1` for equilateral, approaching `0` for a sliver.
///
/// The caller owns the degenerate case: a triangle with no longest side has no
/// quality, and what stands in for one is the caller's own policy.
template <typename T>
auto triangle_quality(T two_area, T max_edge_length2) -> T {
  return tf::two_over_sqrt_3<T> * two_area / max_edge_length2;
}

} // namespace tf
