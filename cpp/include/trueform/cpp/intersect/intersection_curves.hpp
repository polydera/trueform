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
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/intersect/intersect_config.hpp"
#include "trueform/intersect/intersect_mode.hpp"

#include <cstddef>
#include <vector>

namespace tf::cpp {

/// @brief Extract exact pair curves from two 3D meshes at either arity. Mixed
/// index inputs return paths in their lossless common dtype.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto intersection_curves(
    const mesh<Index0, Real, 3, Ngon0> &a,
    const mesh<Index1, Real, 3, Ngon1> &b,
    tf::intersect_config config = {tf::intersect_mode::sos})
    -> tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3>;

/// @brief Exact intersection curves of a range of operands. A range is
/// homogeneous, so its element states the arity the operands are read at.
template <typename Index, typename Real, std::size_t Ngon>
auto intersection_curves(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                         tf::intersect_config config =
                             {tf::intersect_mode::sos |
                              tf::intersect_mode::resolve_crossing_contours})
    -> tf::curves_buffer<Index, Real, 3>;

#define TF_CPP_EXTERN_INTERSECTION_CURVES_PAIR(Index0, Real, Index1, Ngon0,    \
                                               Ngon1)                          \
  extern template auto                                                         \
  intersection_curves<Index0, Real, Index1, Ngon0, Ngon1>(                     \
      const mesh<Index0, Real, 3, Ngon0> &,                                    \
      const mesh<Index1, Real, 3, Ngon1> &, tf::intersect_config)              \
      -> tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3>

#define TF_CPP_EXTERN_INTERSECTION_CURVES_RANGE(Index, Real, Ngon)             \
  extern template auto intersection_curves<Index, Real, Ngon>(                 \
      const std::vector<mesh<Index, Real, 3, Ngon>> &, tf::intersect_config)   \
      -> tf::curves_buffer<Index, Real, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_INDEX_NGON_PAIR(
    TF_CPP_EXTERN_INTERSECTION_CURVES_PAIR)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_INTERSECTION_CURVES_RANGE)

#undef TF_CPP_EXTERN_INTERSECTION_CURVES_RANGE
#undef TF_CPP_EXTERN_INTERSECTION_CURVES_PAIR

} // namespace tf::cpp
