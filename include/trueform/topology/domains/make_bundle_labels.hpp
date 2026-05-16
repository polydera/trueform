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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/buffer.hpp"
#include "../connected_component_labels.hpp"
#include <array>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Group MEL fragments into 3D-connected bundles via union-find on
/// fragment-only merge pairs.
///
/// Two MEL fragments belong to the same bundle iff they are linked by a
/// chain of non-manifold edges (each chain link contributed by
/// @ref tf::topology::domains::emit_domain_merges as a fragment-only
/// pair in the `bundle_merges` buffer). Singleton fragments with no NM
/// incidence get their own bundle id.
///
/// @param bundle_merges Fragment-only `{min, max}` merge pairs.
/// @param n_components Number of MEL fragments.
/// @return @ref tf::connected_component_labels with `labels[c]` =
///         bundle id of fragment `c` and `n_components` = number of
///         distinct bundles.
template <typename Index>
auto make_bundle_labels(const tf::buffer<std::array<Index, 2>> &bundle_merges,
                        Index n_components)
    -> tf::connected_component_labels<Index> {
  tf::connected_component_labels<Index> out;
  out.labels.allocate(n_components);
  out.n_components =
      Index(tf::make_dense_equivalence_class_map(bundle_merges, out.labels));
  return out;
}

} // namespace tf::topology::domains
