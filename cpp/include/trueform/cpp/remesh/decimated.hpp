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
#include "trueform/remesh/decimate_config.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// @brief Decimate a three-dimensional triangle mesh, preserving its index
/// storage.
///
/// A stored transformation supplies the operation's coordinate frame where the
/// underlying algorithm uses one, but is not applied to or transferred onto
/// the result.
/// @note Reads the HALF EDGES and the FACE MEMBERSHIP they stand on.
template <typename Index, typename Real>
auto decimated(const mesh<Index, Real, 3> &value, Real target_proportion,
               tf::decimate_config<Real> config = {},
               const nd_array<std::int32_t> &regions = {})
    -> remesh_result<Index, Real>;

#define TF_CPP_EXTERN_DECIMATED(Index, Real)                                   \
  extern template auto decimated<Index, Real>(                                 \
      const mesh<Index, Real, 3> &, Real, tf::decimate_config<Real>,           \
      const nd_array<std::int32_t> &) -> remesh_result<Index, Real>

/// A remesh is of a three-dimensional triangle mesh by its own definition, so
/// a build without that dimension has none to state.
#if TF_CPP_MATRIX_HAS_3D
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_DECIMATED)
#endif

#undef TF_CPP_EXTERN_DECIMATED

} // namespace tf::cpp
