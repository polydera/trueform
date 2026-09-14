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

#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"

#include <cstdint>

namespace tf::cpp {

/// @brief Create a closed cylinder centered on the z-axis.
///
/// Segments must be at least three.
template <typename Index = default_index_t, typename Real>
auto make_cylinder_mesh(Real radius, Real height, std::int32_t segments) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type;

#define TF_CPP_EXTERN_MAKE_CYLINDER_MESH(Index, Real)                          \
  extern template auto make_cylinder_mesh<Index, Real>(Real, Real,             \
                                                       std::int32_t) ->        \
      typename detail::minted_mesh_result<Index, Real, 3>::type

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_MAKE_CYLINDER_MESH)

#undef TF_CPP_EXTERN_MAKE_CYLINDER_MESH

} // namespace tf::cpp
