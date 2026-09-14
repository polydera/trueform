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

/// @brief Copy a mesh and reverse every face winding, at either arity.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reverse_winding(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_REVERSE_WINDING(Index, Real, Dims, Ngon)                 \
  extern template auto reverse_winding<Index, Real, Dims, Ngon>(               \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_REVERSE_WINDING)

#undef TF_CPP_EXTERN_REVERSE_WINDING

} // namespace tf::cpp
