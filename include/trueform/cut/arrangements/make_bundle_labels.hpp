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
#include "../../topology/connected_component_labels.hpp"
#include "../../topology/domains/make_bundle_labels.hpp"
#include <array>

namespace tf::cut {

/// @ingroup cut
/// @brief Group arrangement components into 3D-connected bundles via
///        union-find on the component-only merge pairs.
///
/// Graph-side alias for @ref tf::topology::domains::make_bundle_labels:
/// purely structural — operates on `bundle_merges + n_components`.
/// Single-sourced implementation, reachable from `tf::cut::` for API
/// consistency.
template <typename Index>
auto make_bundle_labels(const tf::buffer<std::array<Index, 2>> &bundle_merges,
                        Index n_components)
    -> tf::connected_component_labels<Index> {
  return tf::topology::domains::make_bundle_labels(bundle_merges, n_components);
}

} // namespace tf::cut
