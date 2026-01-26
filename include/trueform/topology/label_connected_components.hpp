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

#include "./components/finder.hpp"
#include "./components/sequential_finder.hpp"
#include "./connected_component_labels.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Label connected components with a mask for selective processing.
///
/// Uses union-find algorithm to label connected components. Only elements
/// where `mask[i]` is true are processed. The applier function is used
/// to iterate over neighbors.
///
/// @tparam Index The index type.
/// @tparam Range0 The labels range type.
/// @tparam Range1 The mask range type.
/// @tparam F The applier function type.
/// @param labels Output labels range (must be pre-allocated).
/// @param mask Boolean mask indicating which elements to process.
/// @param applier Function `(id, callback)` that calls callback for each neighbor of id.
/// @param expected_number_of_components Hint for algorithm selection.
/// @return The number of connected components found.
template <typename Index = int, typename Range0, typename Range1, typename F>
auto label_connected_components_masked(Range0 &&labels, const Range1 &mask,
                                const F &applier,
                                Index expected_number_of_components = 2) {
  using label_t = std::decay_t<decltype(labels[0])>;
  if (labels.size() < 5000 || expected_number_of_components > 200) {
    tf::topology::sequential_connected_components_finder<Index, label_t> finder;
    return finder.run(labels, mask, applier);
  } else {
    tf::topology::connected_components_finder<Index, label_t> finder;
    return finder.run(labels, mask, applier);
  }
}

/// @ingroup topology_components
/// @brief Label connected components with mask into a type-safe container.
/// @tparam Index The index type.
/// @tparam LabelType The label type.
/// @tparam Range1 The mask range type.
/// @tparam F The applier function type.
/// @param cl The output container (labels must be pre-allocated).
/// @param mask Boolean mask indicating which elements to process.
/// @param applier Function `(id, callback)` that calls callback for each neighbor of id.
/// @param expected_number_of_components Hint for algorithm selection.
template <typename Index = int, typename LabelType, typename Range1, typename F>
auto label_connected_components_masked(tf::connected_component_labels<LabelType> &cl,
                                const Range1 &mask, const F &applier,
                                Index expected_number_of_components = 2) {
  cl.n_components = label_connected_components_masked(
      cl.labels, mask, applier, expected_number_of_components);
}

/// @ingroup topology_components
/// @brief Label all connected components.
///
/// Uses union-find algorithm to label connected components. The applier
/// function defines the connectivity (which elements are neighbors).
///
/// @tparam Index The index type.
/// @tparam Range The labels range type.
/// @tparam F The applier function type.
/// @param labels Output labels range (must be pre-allocated).
/// @param applier Function `(id, callback)` that calls callback for each neighbor of id.
/// @param expected_number_of_components Hint for algorithm selection.
/// @return The number of connected components found.
template <typename Index = int, typename Range, typename F>
auto label_connected_components(Range &&labels, const F &applier,
                                Index expected_number_of_components = 2) {
  return label_connected_components_masked(
      labels, tf::make_constant_range(true, labels.size()), applier,
      expected_number_of_components);
}

/// @ingroup topology_components
/// @brief Label all connected components into a type-safe container.
/// @tparam Index The index type.
/// @tparam LabelType The label type.
/// @tparam F The applier function type.
/// @param cl The output container (labels must be pre-allocated).
/// @param applier Function `(id, callback)` that calls callback for each neighbor of id.
/// @param expected_number_of_components Hint for algorithm selection.
template <typename Index = int, typename LabelType, typename F>
auto label_connected_components(tf::connected_component_labels<LabelType> &cl,
                                const F &applier,
                                Index expected_number_of_components = 2) {
  cl.n_components = label_connected_components_masked(cl.labels, applier,
                                               expected_number_of_components);
}
} // namespace tf
