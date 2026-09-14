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

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Compute the signed volume of a closed mesh in its transformed frame.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto signed_volume(const mesh<Index, Real, Dims, Ngon> &value) -> Real;

#define TF_CPP_EXTERN_SIGNED_VOLUME(Index, Real, Ngon)                         \
  extern template auto signed_volume<Index, Real, 3, Ngon>(                    \
      const mesh<Index, Real, 3, Ngon> &) -> Real

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_SIGNED_VOLUME)

#undef TF_CPP_EXTERN_SIGNED_VOLUME

} // namespace tf::cpp
