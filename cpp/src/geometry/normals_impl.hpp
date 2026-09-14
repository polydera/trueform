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

#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/geometry/point_normals.hpp"

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// The cache is the one producer of a mesh's normals; an entry that hands them
/// to a caller hands a COPY, so writing through the result cannot reach the
/// values every other reading of this geometry shares.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto normals(const mesh<Index, Real, Dims, Ngon> &value) -> nd_array<Real> {
  return value.cache().face_normals_handle(value.geometry()).deep_copy();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto point_normals(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<Real> {
  return value.cache().point_normals_handle(value.geometry()).deep_copy();
}

} // namespace tf::cpp
