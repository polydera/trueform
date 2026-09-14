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

#include <cmath>

namespace tf::geometry {

/// @brief Gaussian weight of one k-NN correspondence, anchored on the nearest.
///
/// The kernel reads the neighbor's excess over the nearest metric, so the
/// nearest neighbor weighs exactly 1 and the anchor cancels when the weighted
/// sum is normalized. The sum is therefore never zero: a kernel width that
/// underflows collapses the correspondence onto the nearest neighbor, which is
/// the kernel's own limit.
///
/// @param metric Squared distance of this neighbor.
/// @param nearest_metric Squared distance of the nearest neighbor.
/// @param sig Squared kernel width (non-negative).
/// @return The neighbor's unnormalized weight.
template <typename T>
auto knn_correspondence_weight(T metric, T nearest_metric, T sig) -> T {
  const T excess = metric - nearest_metric;
  return excess > T(0) ? std::exp(-excess / (T(2) * sig)) : T(1);
}

} // namespace tf::geometry
