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

/// @ingroup geometry_alignment
/// @brief Configuration for ICP (Iterative Closest Point) alignment.
struct icp_config {
  /// Maximum number of iterations.
  std::size_t max_iterations = 100;

  /// Stop when relative improvement falls below this (e.g., 1e-6 = 0.0001%).
  float min_relative_improvement = 1e-6f;

  /// EMA smoothing factor for error (0-1, higher = less smoothing).
  float ema_alpha = 0.3f;

  /// Number of points to subsample (0 = use all points).
  std::size_t n_samples = 1000;

  /// Number of nearest neighbors for soft correspondences.
  std::size_t k = 1;

  /// Gaussian kernel width for soft correspondences (-1 = adaptive).
  float sigma = -1.f;

  /// Proportion of worst correspondences to reject.
  float outlier_proportion = 0.f;
};

} // namespace tf
