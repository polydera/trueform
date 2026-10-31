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

/// @brief The `(sheet tag, volume tag)` pairs whose volume holds a free rim of
/// the sheet, ascending, shape `(K, 2)`: every domain inside reads no side of
/// that sheet.
template <typename Index, typename Real>
auto csg_sheet_rim_volumes(const csg_graph<Index, Real> &graph)
    -> nd_array<Index>;

#define TF_CPP_EXTERN_CSG_SHEET_RIM_VOLUMES(Index, Real)                       \
  extern template auto csg_sheet_rim_volumes(const csg_graph<Index, Real> &)   \
      -> nd_array<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_SHEET_RIM_VOLUMES)

#undef TF_CPP_EXTERN_CSG_SHEET_RIM_VOLUMES

} // namespace tf::cpp
