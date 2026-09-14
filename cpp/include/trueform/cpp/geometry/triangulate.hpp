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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Triangulate a typed fixed- or dynamic-connectivity mesh.
///
/// The result always has fixed triangle connectivity, owns independent face
/// and point arrays, and does not inherit or apply the input transformation:
/// the mesh is read in its raw point storage.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto triangulate(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, 3>;

/// @brief Triangulate typed dynamic polygon faces over Dims-dimensional points.
template <typename Index, typename Real, std::size_t Dims = 3>
auto triangulate(const offset_blocked_buffer<Index, Index> &faces,
                 const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3>;

/// @brief Triangulate typed fixed-width polygon faces over Dims-dimensional
/// points. Width-three input is copied exactly.
template <typename Index, typename Real, std::size_t Dims = 3>
auto triangulate(const nd_array<Index> &faces, const nd_array<Real> &points)
    -> tf::polygons_buffer<Index, Real, Dims, 3>;

/// @brief Triangulate one [V, Dims] polygon or a [N, V, Dims] polygon batch.
template <typename Index, typename Real, std::size_t Dims = 3>
auto triangulate(const nd_array<Real> &polygons)
    -> tf::polygons_buffer<Index, Real, Dims, 3>;

#define TF_CPP_EXTERN_TRIANGULATE_AT_ARITY(Index, Real, Dims, Ngon)            \
  extern template auto triangulate<Index, Real, Dims, Ngon>(                   \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> tf::polygons_buffer<Index, Real, Dims, 3>

#define TF_CPP_EXTERN_TRIANGULATE_MATRIX(Index, Real, Dims)                    \
  extern template auto triangulate<Index, Real, Dims>(                         \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &)     \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  extern template auto triangulate<Index, Real, Dims>(const nd_array<Index> &, \
                                                      const nd_array<Real> &)  \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  extern template auto triangulate<Index, Real, Dims>(const nd_array<Real> &)  \
      -> tf::polygons_buffer<Index, Real, Dims, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_TRIANGULATE_AT_ARITY)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_EXTERN_TRIANGULATE_MATRIX)

#undef TF_CPP_EXTERN_TRIANGULATE_MATRIX
#undef TF_CPP_EXTERN_TRIANGULATE_AT_ARITY

} // namespace tf::cpp
