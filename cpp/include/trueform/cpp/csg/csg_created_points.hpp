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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"

namespace tf::cpp {

/// @brief The points the arrangement minted, on the graph's own lattice.
template <typename Index, typename Real>
auto csg_created_points(const csg_graph<Index, Real> &graph) -> nd_array<Real>;

#define TF_CPP_EXTERN_CSG_CREATED_POINTS(Index, Real)                          \
  extern template auto csg_created_points(const csg_graph<Index, Real> &)      \
      -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_CREATED_POINTS)

#undef TF_CPP_EXTERN_CSG_CREATED_POINTS

} // namespace tf::cpp
