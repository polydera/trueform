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
#include "trueform/cpp/remesh/remesh_result.hpp"
#include "trueform/remesh/isotropic_remesh_config.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// @brief Isotropically remesh a supported three-dimensional fixed triangle
/// mesh while preserving its index storage.
/// @note Reads the HALF EDGES and the FACE MEMBERSHIP they stand on.
template <typename Index, typename Real>
auto isotropic_remeshed(const mesh<Index, Real, 3> &value,
                        tf::isotropic_remesh_config<Real> config,
                        const nd_array<std::int32_t> &regions = {})
    -> remesh_result<Index, Real>;

#define TF_CPP_EXTERN_ISOTROPIC_REMESHED(Index, Real)                          \
  extern template auto isotropic_remeshed<Index, Real>(                        \
      const mesh<Index, Real, 3> &, tf::isotropic_remesh_config<Real>,         \
      const nd_array<std::int32_t> &) -> remesh_result<Index, Real>

/// A remesh is of a three-dimensional triangle mesh by its own definition, so
/// a build without that dimension has none to state.
#if TF_CPP_MATRIX_HAS_3D
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_ISOTROPIC_REMESHED)
#endif

#undef TF_CPP_EXTERN_ISOTROPIC_REMESHED

} // namespace tf::cpp
