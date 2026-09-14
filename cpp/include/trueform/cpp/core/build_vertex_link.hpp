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

#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Fill a mesh's vertex link, if it is missing or stale.
///
/// The build verbs' contract, and what a caller names when it prebuilds — see
/// `trueform/cpp/core/build_tree.hpp`.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto build_vertex_link(const mesh<Index, Real, Dims, Ngon> &value) -> void {
  static_cast<void>(value.vertex_link());
}

/// @overload Fill an edge mesh's vertex link.
template <typename Index, typename Real, std::size_t Dims>
auto build_vertex_link(const edge_mesh<Index, Real, Dims> &value) -> void {
  static_cast<void>(value.vertex_link());
}

} // namespace tf::cpp
