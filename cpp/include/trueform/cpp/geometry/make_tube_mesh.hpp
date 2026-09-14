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
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstdint>

namespace tf::cpp {

/// @brief Create tube meshes around a valid collection of 3D curves.
///
/// The curves are the two arrays a polyline is: the paths, blocked by curve,
/// and the `[N, 3]` points they name. Radial segments must be at least three.
/// Curves with fewer than two points are preserved as valid empty
/// contributions to the result.
template <typename Index, typename Real>
auto make_tube_mesh(const offset_blocked_buffer<Index, Index> &paths,
                    const nd_array<Real> &points, Real radius,
                    std::int32_t radial_segments = 8) ->
    typename detail::minted_mesh_result<Index, Real, 3>::type;

#define TF_CPP_EXTERN_MAKE_TUBE_MESH(Index, Real)                              \
  extern template auto make_tube_mesh<Index, Real>(                            \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      Real, std::int32_t) ->                                                   \
      typename detail::minted_mesh_result<Index, Real, 3>::type

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_MAKE_TUBE_MESH)

#undef TF_CPP_EXTERN_MAKE_TUBE_MESH

} // namespace tf::cpp
