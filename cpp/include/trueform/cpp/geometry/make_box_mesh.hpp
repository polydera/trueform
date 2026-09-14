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

/// @brief Create an axis-aligned box centered at the origin.
template <typename Index = default_index_t, typename Real>
auto make_box_mesh(Real width, Real height, Real depth) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type;

/// @brief Create a subdivided axis-aligned box centered at the origin.
///
/// Every tick count must be positive.
template <typename Index = default_index_t, typename Real>
auto make_box_mesh(Real width, Real height, Real depth,
                   std::int32_t width_ticks, std::int32_t height_ticks,
                   std::int32_t depth_ticks) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type;

#define TF_CPP_EXTERN_MAKE_BOX_MESH(Index, Real)                               \
  extern template auto make_box_mesh<Index, Real>(Real, Real, Real)            \
      ->typename detail::minted_mesh_result<Index, Real, 3>::type;             \
  extern template auto make_box_mesh<Index, Real>(                             \
      Real, Real, Real, std::int32_t, std::int32_t, std::int32_t) ->           \
      typename detail::minted_mesh_result<Index, Real, 3>::type

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_MAKE_BOX_MESH)

#undef TF_CPP_EXTERN_MAKE_BOX_MESH

} // namespace tf::cpp
