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
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Compute unit normals for a runtime 3D triangle or polygon.
///
/// The first three vertices define each normal. A single primitive returns an
/// array with shape [3], while a batch returns [N, 3], including [0, 3].
template <typename Real>
auto normals(const primitive<Real, 3> &value) -> nd_array<Real>;

/// @brief Compute the unit face normals of a 3D mesh.
///
/// The normals are read from the mesh's raw stored points; its frame is not
/// applied, so the array names one value per face of the mesh itself. The
/// cache remembers them and this hands back a COPY, so writing through the
/// result reaches nothing another reading shares.
/// @note Reads the FACE NORMALS.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto normals(const mesh<Index, Real, Dims, Ngon> &value) -> nd_array<Real>;

#define TF_CPP_EXTERN_PRIMITIVE_NORMALS(Real)                                  \
  extern template auto normals<Real>(const primitive<Real, 3> &)               \
      -> nd_array<Real>

#define TF_CPP_EXTERN_MESH_NORMALS(Index, Real, Ngon)                          \
  extern template auto normals<Index, Real, 3, Ngon>(                          \
      const mesh<Index, Real, 3, Ngon> &) -> nd_array<Real>

/// A normal is of a three-dimensional surface by its own definition, so a
/// build without that dimension has none to state.
#if TF_CPP_MATRIX_HAS_3D
TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_EXTERN_PRIMITIVE_NORMALS)
#endif
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_MESH_NORMALS)

#undef TF_CPP_EXTERN_MESH_NORMALS
#undef TF_CPP_EXTERN_PRIMITIVE_NORMALS

} // namespace tf::cpp
