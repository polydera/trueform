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

#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Fill a mesh's manifold edge link, if it is missing or stale.
///
/// The build verbs' contract, and what a caller names when it prebuilds — see
/// `trueform/cpp/core/build_tree.hpp`.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto build_manifold_edge_link(const mesh<Index, Real, Dims, Ngon> &value)
    -> void {
  static_cast<void>(value.manifold_edge_link());
}

} // namespace tf::cpp
