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

/// @brief Principal curvature values at every mesh vertex.
template <typename Real> struct principal_curvatures_result {
  nd_array<Real> k0;
  nd_array<Real> k1;
};

/// @brief Compute principal curvature values using a non-negative k-ring
/// neighborhood. A zero-ring neighborhood is supported.
/// @note Reads the POINT NORMALS, the FACE MEMBERSHIP and the VERTEX LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto principal_curvatures(const mesh<Index, Real, Dims, Ngon> &value, int k = 2)
    -> principal_curvatures_result<Real>;

#define TF_CPP_EXTERN_PRINCIPAL_CURVATURES(Index, Real, Ngon)                  \
  extern template auto principal_curvatures<Index, Real, 3, Ngon>(             \
      const mesh<Index, Real, 3, Ngon> &, int)                                 \
      -> principal_curvatures_result<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_PRINCIPAL_CURVATURES)

#undef TF_CPP_EXTERN_PRINCIPAL_CURVATURES

} // namespace tf::cpp
