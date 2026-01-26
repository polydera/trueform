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
#include "../core/buffer.hpp"
namespace tf {

/// @ingroup topology_components
/// @brief Type-safe container for connected component labels.
///
/// Stores per-element component labels and the total number of components.
/// Used by @ref tf::label_connected_components() and related functions.
///
/// @tparam LabelType The integer type for component labels.
template <typename LabelType> struct connected_component_labels {
  /// @brief Per-element component labels (0-indexed).
  tf::buffer<LabelType> labels;
  /// @brief Total number of connected components.
  LabelType n_components;
};
} // namespace tf
