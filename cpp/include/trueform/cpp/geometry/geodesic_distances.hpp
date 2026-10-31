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

#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf::cpp {

/// @brief The geodesic distance from the nearest seed to every point: zero at
/// a seed, infinity where no front reaches or past `radius`, at most `radius`
/// otherwise, in the mesh's own type. Measured in the stored coordinates, at
/// identity.
/// @note Reads the FACE MEMBERSHIP and the VERTEX LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto geodesic_distances(const mesh<Index, Real, Dims, Ngon> &value,
                        const nd_array<Index> &seeds,
                        double radius = std::numeric_limits<double>::infinity())
    -> nd_array<Real>;

#define TF_CPP_EXTERN_GEODESIC_DISTANCES(Index, Real, Ngon)                    \
  extern template auto geodesic_distances<Index, Real, 3, Ngon>(               \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Index> &, double)     \
      -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_GEODESIC_DISTANCES)

#undef TF_CPP_EXTERN_GEODESIC_DISTANCES

} // namespace tf::cpp
