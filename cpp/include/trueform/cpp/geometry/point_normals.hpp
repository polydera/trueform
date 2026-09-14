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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Compute the unit point normals of a 3D mesh.
///
/// A point normal is the average of its adjacent unit face normals. The array
/// names one value per point. The cache remembers them and this hands back a
/// COPY, so writing through the result reaches nothing another reading shares.
/// @note Reads the POINT NORMALS, which stand on the face normals and the
/// FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto point_normals(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<Real>;

#define TF_CPP_EXTERN_POINT_NORMALS(Index, Real, Ngon)                         \
  extern template auto point_normals<Index, Real, 3, Ngon>(                    \
      const mesh<Index, Real, 3, Ngon> &) -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_POINT_NORMALS)

#undef TF_CPP_EXTERN_POINT_NORMALS

} // namespace tf::cpp
