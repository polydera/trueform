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

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {
/// @brief The boundary paths and the points they name, densely renumbered.
template <typename Index, typename Real, std::size_t Dims = 3>
struct boundary_curves_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "boundary_curves_result requires an unqualified supported "
                "index type");
  static_assert(Dims == 2 || Dims == 3,
                "boundary_curves_result Dims must be 2 or 3");

  offset_blocked_buffer<Index, Index> paths;
  nd_array<Real> points;
};

/// @brief The boundary paths and the points they name.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_curves(const mesh<Index, Real, Dims, Ngon> &value)
    -> boundary_curves_result<Index, Real, Dims>;

#define TF_CPP_EXTERN_BOUNDARY_CURVES(Index, Real, Dims, Ngon)                 \
  extern template auto boundary_curves<Index, Real, Dims, Ngon>(               \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> boundary_curves_result<Index, Real, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_BOUNDARY_CURVES)

#undef TF_CPP_EXTERN_BOUNDARY_CURVES

} // namespace tf::cpp
