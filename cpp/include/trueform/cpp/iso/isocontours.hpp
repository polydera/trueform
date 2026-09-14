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

#include "trueform/core/curves_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Extract one raw-coordinate isocontour from a 3D mesh at either
/// arity, retaining its path-index dtype.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars, Real cut_value)
    -> tf::curves_buffer<Index, Real, 3>;

/// @brief Extract raw-coordinate isocontours for a threshold array from a 3D
/// mesh at either arity, retaining its path-index dtype.
template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars,
                 const nd_array<Real> &cut_values)
    -> tf::curves_buffer<Index, Real, 3>;

#define TF_CPP_EXTERN_ISOCONTOURS_3D(Index, Real, Ngon)                        \
  extern template auto isocontours<Index, Real, Ngon>(                         \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &, Real)        \
      -> tf::curves_buffer<Index, Real, 3>;                                    \
  extern template auto isocontours<Index, Real, Ngon>(                         \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &,              \
      const nd_array<Real> &) -> tf::curves_buffer<Index, Real, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_ISOCONTOURS_3D)

#undef TF_CPP_EXTERN_ISOCONTOURS_3D

} // namespace tf::cpp
