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
#include <type_traits>

namespace tf::cpp {

/// @brief Fill a mesh's winding moments, and the tree they mirror, if either is
/// missing or stale.
///
/// The build verbs' contract, and what a caller names when it prebuilds — see
/// `trueform/cpp/core/build_tree.hpp`. It is what `signed_distance` names, and
/// a winding number is of a three-dimensional surface.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto build_winding_moments(const mesh<Index, Real, Dims, Ngon> &value) -> void {
  static_cast<void>(value.winding_moments());
}

} // namespace tf::cpp
