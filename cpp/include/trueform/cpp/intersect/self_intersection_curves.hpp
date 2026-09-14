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
#include "trueform/intersect/intersect_config.hpp"
#include "trueform/intersect/intersect_mode.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Exact self-intersection curves of one 3D mesh at either arity.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto self_intersection_curves(
    const mesh<Index, Real, 3, Ngon> &value,
    tf::intersect_config config = {tf::intersect_mode::sos |
                                   tf::intersect_mode::resolve_contours})
    -> tf::curves_buffer<Index, Real, 3>;

#define TF_CPP_EXTERN_SELF_INTERSECTION_CURVES(Index, Real, Ngon)              \
  extern template auto self_intersection_curves<Index, Real, Ngon>(            \
      const mesh<Index, Real, 3, Ngon> &, tf::intersect_config)                \
      -> tf::curves_buffer<Index, Real, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_SELF_INTERSECTION_CURVES)

#undef TF_CPP_EXTERN_SELF_INTERSECTION_CURVES

} // namespace tf::cpp
