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
#include "trueform/cpp/fill/hole_fill_report.hpp"
#include "trueform/fill/hole_fill_config.hpp"

namespace tf::cpp {

/// @brief Fill every boundary rim of a three-dimensional triangle mesh.
///
/// Group `g` fills rim `g` of `boundary_rims`, and a rim the fill refuses
/// leaves the others alone. Geometry is read in the raw local frame, so a
/// stated placement is not applied.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value,
                tf::hole_fill_config config = {})
    -> hole_fill_report<Index, Real>;

/// @brief Fill the named boundary rims alone.
///
/// `rim_ids` indexes the rims `boundary_rims` states, and group `g` fills
/// `rim_ids[g]`. An id out of range, a repeated one, or one naming an open rim
/// is refused.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value,
                const nd_array<Index> &rim_ids,
                tf::hole_fill_config config = {})
    -> hole_fill_report<Index, Real>;

#define TF_CPP_EXTERN_FILL_HOLES(Index, Real)                                  \
  extern template auto fill_holes<Index, Real>(const mesh<Index, Real, 3> &,   \
                                               tf::hole_fill_config)           \
      -> hole_fill_report<Index, Real>;                                        \
  extern template auto fill_holes<Index, Real>(                                \
      const mesh<Index, Real, 3> &, const nd_array<Index> &,                   \
      tf::hole_fill_config) -> hole_fill_report<Index, Real>

/// A fill mints three-dimensional points by its own definition, so a build
/// without that dimension has none to state.
#if TF_CPP_MATRIX_HAS_3D
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_FILL_HOLES)
#endif

#undef TF_CPP_EXTERN_FILL_HOLES

} // namespace tf::cpp
