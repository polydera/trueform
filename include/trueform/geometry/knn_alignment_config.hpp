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

#include <cstddef>

namespace tf {

/// @ingroup geometry_registration
/// @brief Configuration for k-NN alignment.
///
/// Controls the behavior of @ref tf::fit_knn_alignment.
///
/// @see @ref tf::fit_knn_alignment
struct knn_alignment_config {
  /// Number of nearest neighbors for soft correspondences.
  /// k=1 is classic ICP (single nearest neighbor).
  /// k>1 uses Gaussian-weighted soft correspondences for robustness.
  std::size_t k = 1;

  /// Gaussian kernel width for soft correspondences.
  /// If negative, uses the k-th neighbor distance (adaptive scaling).
  float sigma = -1.f;

  /// Proportion of worst correspondences to reject (0 to 1).
  /// 0 = no rejection, 0.1 = reject worst 10%.
  /// Provides robustness to partial overlap and outliers.
  float outlier_proportion = 0.f;
};

} // namespace tf
