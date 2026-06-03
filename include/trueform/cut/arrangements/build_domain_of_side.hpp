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
#include "../../topology/domains/build_domain_of_side.hpp"
#include <array>

namespace tf::cut {

/// @ingroup cut
/// @brief Run union-find on the component-side merge pairs to obtain
///        the per-component-side domain id buffer.
///
/// Graph-side alias for @ref tf::topology::domains::build_domain_of_side:
/// purely structural — operates on merges + `2 * n_components`. Same
/// single-sourced implementation, reachable from `tf::cut::` for API
/// consistency.
template <typename Index>
auto build_domain_of_side(const tf::buffer<std::array<Index, 2>> &merges,
                          Index n_components,
                          tf::buffer<Index> &domain_of_side) -> Index {
  return tf::topology::domains::build_domain_of_side(merges, n_components,
                                                      domain_of_side);
}

} // namespace tf::cut
