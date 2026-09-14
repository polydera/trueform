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

#include <cstddef>

namespace tf::cpp {
/// @brief The same geometry, in storage of its own, consistently wound.
///
/// The winding is rewritten, so the mesh is copied and the copy is what is
/// oriented, against the source mesh's own edge link. The frame is the
/// reading's and does not travel with the coordinates.
/// @note Reads the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto orient_faces_consistently(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_ORIENT_FACES_CONSISTENTLY(Index, Real, Dims, Ngon)       \
  extern template auto orient_faces_consistently<Index, Real, Dims, Ngon>(     \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(
    TF_CPP_EXTERN_ORIENT_FACES_CONSISTENTLY)

#undef TF_CPP_EXTERN_ORIENT_FACES_CONSISTENTLY

} // namespace tf::cpp
