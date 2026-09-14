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

/// @brief Return a mesh whose points have undergone Taubin smoothing.
/// Zero iterations preserve the input point values.
/// @note Reads the VERTEX LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto taubin_smoothed(const mesh<Index, Real, Dims, Ngon> &value, int iterations,
                     Real lambda = Real{0.5}, Real kpb = Real{0.1})
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_TAUBIN_SMOOTHED(Index, Real, Dims, Ngon)                 \
  extern template auto taubin_smoothed<Index, Real, Dims, Ngon>(               \
      const mesh<Index, Real, Dims, Ngon> &, int, Real, Real)                  \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_TAUBIN_SMOOTHED)

#undef TF_CPP_EXTERN_TAUBIN_SMOOTHED

} // namespace tf::cpp
