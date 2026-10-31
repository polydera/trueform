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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/geometry/dirichlet_field_status.hpp"
#include "trueform/geometry/dirichlet_order.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief One value per point, in the mesh's own type, and what its system
/// answered.
template <typename Real> struct dirichlet_field_result {
  nd_array<Real> field;
  tf::dirichlet_field_status status;
};

/// @brief The field held at `fixed_ids` and vanishing under `order`'s
/// operator elsewhere, over the Laplacian the corner weights state:
/// `corner_weights[3 f + c]` weighs the edge of face `f` opposite its corner
/// `c`. The solve is sparse Cholesky; a refused `status` names why, and its
/// field states the fixed values alone. Read at identity.
template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<Dims == 3, int> = 0>
auto dirichlet_field(const mesh<Index, Real, Dims, 3> &value,
                     const nd_array<double> &corner_weights,
                     const nd_array<Index> &fixed_ids,
                     const nd_array<double> &fixed_values,
                     tf::dirichlet_order order) -> dirichlet_field_result<Real>;

/// @overload The cotangent weights.
template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<Dims == 3, int> = 0>
auto dirichlet_field(const mesh<Index, Real, Dims, 3> &value,
                     const nd_array<Index> &fixed_ids,
                     const nd_array<double> &fixed_values,
                     tf::dirichlet_order order) -> dirichlet_field_result<Real>;

#define TF_CPP_EXTERN_DIRICHLET_FIELD(Index, Real)                             \
  extern template auto dirichlet_field<Index, Real, 3>(                        \
      const mesh<Index, Real, 3, 3> &, const nd_array<double> &,               \
      const nd_array<Index> &, const nd_array<double> &, tf::dirichlet_order)  \
      -> dirichlet_field_result<Real>;                                         \
  extern template auto dirichlet_field<Index, Real, 3>(                        \
      const mesh<Index, Real, 3, 3> &, const nd_array<Index> &,                \
      const nd_array<double> &, tf::dirichlet_order)                           \
      -> dirichlet_field_result<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_DIRICHLET_FIELD)

#undef TF_CPP_EXTERN_DIRICHLET_FIELD

} // namespace tf::cpp
