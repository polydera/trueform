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
#include "trueform/cpp/csg/csg_graph.hpp"

namespace tf::cpp {

/// @brief The cross-tag seam polylines read off the graph.
template <typename Index, typename Real>
auto csg_intersection_curves(const csg_graph<Index, Real> &graph)
    -> tf::curves_buffer<Index, Real, 3>;

#define TF_CPP_EXTERN_CSG_INTERSECTION_CURVES(Index, Real)                     \
  extern template auto csg_intersection_curves(const csg_graph<Index, Real> &) \
      -> tf::curves_buffer<Index, Real, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_INTERSECTION_CURVES)

#undef TF_CPP_EXTERN_CSG_INTERSECTION_CURVES

} // namespace tf::cpp
