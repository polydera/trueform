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
#include <array>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Run union-find on the fragment-side merge pairs to obtain
/// the per-fragment-side domain id buffer.
///
/// The union-find half of the domain-extraction pipeline, separated
/// from the per-face fan-out so downstream stages (per-domain volume,
/// nesting) can read `domain_of_side` and emit further merges before
/// the per-face write is paid.
///
/// @param merges Fragment-side merge pairs accumulated by
///        @ref tf::topology::domains::emit_boundary_merges and
///        @ref tf::topology::domains::emit_domain_merges (and any
///        synthetic anchor merges from Mode 1).
/// @param n_components Number of MEL fragments; `domain_of_side` is
///        sized `2 * n_components`.
/// @param domain_of_side Output buffer filled with the per-fragment-side
///        domain id (allocated and written by this function).
/// @return The number of distinct domains produced by union-find.
template <typename Index>
auto build_domain_of_side(const tf::buffer<std::array<Index, 2>> &merges,
                          Index n_components,
                          tf::buffer<Index> &domain_of_side) -> Index {
  domain_of_side.allocate(2 * n_components);
  return Index(tf::make_dense_equivalence_class_map(merges, domain_of_side));
}

} // namespace tf::topology::domains
