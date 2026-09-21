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

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief The edges two faces share, shaped [N, 2], and the radians they turn
/// through, shaped [N] and aligned with them.
template <typename Index, typename Real> struct dihedral_angles_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "dihedral_angles_result requires an unqualified supported "
                "index type");
  nd_array<Index> edges;
  nd_array<Real> angles;
};

/// @brief Measure every edge two faces of a mesh share.
///
/// One entry per undirected edge, reached through the manifold edge link's
/// representative. A boundary or non-manifold edge joins no pair of faces and
/// turns through no angle, so it is not stated. The normals are read off the
/// stored points; the mesh's frame is not applied.
/// @note Reads the MANIFOLD EDGE LINK and the FACE NORMALS.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto dihedral_angles(const mesh<Index, Real, Dims, Ngon> &value)
    -> dihedral_angles_result<Index, Real>;

#define TF_CPP_EXTERN_DIHEDRAL_ANGLES(Index, Real, Ngon)                       \
  extern template auto dihedral_angles<Index, Real, 3, Ngon>(                  \
      const mesh<Index, Real, 3, Ngon> &) -> dihedral_angles_result<Index, Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_DIHEDRAL_ANGLES)

#undef TF_CPP_EXTERN_DIHEDRAL_ANGLES

} // namespace tf::cpp
