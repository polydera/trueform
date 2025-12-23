/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./connected_component_labels.hpp"
#include "./set_type.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Component labels with per-component set type information.
///
/// Extends @ref tf::connected_component_labels with additional metadata
/// indicating the type of each component (e.g., inner vs outer boundary).
///
/// @tparam LabelType The integer type for component labels.
template <typename LabelType> struct set_component_labels {
  /// @brief The underlying component labels.
  tf::connected_component_labels<LabelType> component_labels;
  /// @brief Per-component set type classification.
  tf::buffer<tf::set_type> set_types;
};
} // namespace tf
