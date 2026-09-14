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
#include <type_traits>

namespace tf::cpp {

/// @brief Copy a supported three-dimensional mesh and orient its faces with
/// positive outward winding while preserving its index and connectivity
/// carrier.
/// @note Reads the MANIFOLD EDGE LINK unless the winding is already stated
/// consistent.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto positively_oriented(const mesh<Index, Real, Dims, Ngon> &value,
                         bool is_consistent = false)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_POSITIVELY_ORIENTED(Index, Real, Ngon)                   \
  extern template auto positively_oriented<Index, Real, 3, Ngon>(              \
      const mesh<Index, Real, 3, Ngon> &, bool)                                \
      -> tf::polygons_buffer<Index, Real, 3, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_POSITIVELY_ORIENTED)

#undef TF_CPP_EXTERN_POSITIVELY_ORIENTED

} // namespace tf::cpp
