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
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>

namespace tf::cpp {
/// @brief The boundary edges assembled into continuous paths.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_paths(const mesh<Index, Real, Dims, Ngon> &value)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_BOUNDARY_PATHS(Index, Real, Dims, Ngon)                  \
  extern template auto boundary_paths<Index, Real, Dims, Ngon>(                \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_BOUNDARY_PATHS)

#undef TF_CPP_EXTERN_BOUNDARY_PATHS

} // namespace tf::cpp
