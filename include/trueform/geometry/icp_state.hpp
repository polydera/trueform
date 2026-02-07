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

#include "./knn_alignment_state.hpp"

namespace tf {

/// @ingroup geometry_alignment
/// @brief Workspace for ICP (Iterative Closest Point) alignment.
///
/// Alias to the underlying kNN alignment state. Subsampling uses lazy strided
/// views, so no additional index buffers are needed.
template <typename Policy0, typename Policy1>
using icp_state = knn_alignment_state<Policy0, Policy1>;

/// @brief Factory function to create ICP state.
///
/// @param X Source point set (policies inspected but not stored).
/// @param Y Target point set (policies inspected but not stored).
/// @return ICP state struct for the given point set policies.
template <typename Policy0, typename Policy1>
auto make_icp_state(const tf::points<Policy0> &X,
                    const tf::points<Policy1> &Y) {
  return make_knn_alignment_state(X, Y);
}

} // namespace tf
