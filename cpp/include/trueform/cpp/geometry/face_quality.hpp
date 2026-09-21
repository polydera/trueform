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
#include <type_traits>

namespace tf::cpp {

/// @brief One [F] array per measure, each indexed by face.
///
/// The angles are radians. An angle is unsigned and lies in `[0, pi]`, so a
/// reflex corner of a non-convex face reads its explement. `aspect_ratio` is
/// the longest side over the shortest, infinite where a side has no length.
/// `quality` is the triangle measure, `1` for equilateral and approaching `0`
/// for a sliver; a face that is not a triangle has none and reads `-1`.
template <typename Real> struct face_quality_result {
  nd_array<Real> quality;
  nd_array<Real> min_angle;
  nd_array<Real> max_angle;
  nd_array<Real> aspect_ratio;
};

/// @brief Measure every face of a mesh.
///
/// The faces are read in their stored points; the mesh's frame is not applied.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto face_quality(const mesh<Index, Real, Dims, Ngon> &value)
    -> face_quality_result<Real>;

#define TF_CPP_EXTERN_FACE_QUALITY(Index, Real, Ngon)                          \
  extern template auto face_quality<Index, Real, 3, Ngon>(                     \
      const mesh<Index, Real, 3, Ngon> &) -> face_quality_result<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_FACE_QUALITY)

#undef TF_CPP_EXTERN_FACE_QUALITY

} // namespace tf::cpp
