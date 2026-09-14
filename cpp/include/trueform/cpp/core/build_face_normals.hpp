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

/// @brief Fill a mesh's face normals, if they are missing or stale.
///
/// The build verbs' contract, and what a caller names when it prebuilds — see
/// `trueform/cpp/core/build_tree.hpp`. A normal is of a three-dimensional
/// surface by its own definition, so the refusal is this entry's own
/// substitution.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto build_face_normals(const mesh<Index, Real, Dims, Ngon> &value) -> void {
  static_cast<void>(value.face_normals());
}

} // namespace tf::cpp
