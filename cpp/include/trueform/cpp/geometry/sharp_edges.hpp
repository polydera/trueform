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

#include "trueform/core/angle.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Extract manifold mesh edges whose dihedral angle meets the threshold.
///
/// The result has shape [N, 2] and the mesh's own index width. The
/// manifold-edge link is asked of the structure, so a second call reuses it.
/// @note Reads the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto sharp_edges(const mesh<Index, Real, Dims, Ngon> &value,
                 tf::rad<Real> angle_threshold) -> nd_array<Index>;

/// @overload
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto sharp_edges(const mesh<Index, Real, Dims, Ngon> &value,
                 tf::deg<Real> angle_threshold) -> nd_array<Index> {
  return cpp::sharp_edges(value, tf::rad<Real>{angle_threshold});
}

#define TF_CPP_EXTERN_SHARP_EDGES(Index, Real, Ngon)                           \
  extern template auto sharp_edges<Index, Real, 3, Ngon>(                      \
      const mesh<Index, Real, 3, Ngon> &, tf::rad<Real>) -> nd_array<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_SHARP_EDGES)

#undef TF_CPP_EXTERN_SHARP_EDGES

} // namespace tf::cpp
