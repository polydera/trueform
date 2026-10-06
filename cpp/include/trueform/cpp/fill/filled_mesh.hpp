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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/fill/hole_fill_report.hpp"

namespace tf::cpp {

/// @brief The mesh a fill states: the input with its patches, every carrying
/// face a split reached replaced by its plan.
///
/// The report's `result` is replayed as the fill stated it, so `value` is the
/// mesh the report was stated against; a report holding no fill, or one of
/// another point or face count, is refused. The points are the mesh's own and
/// then the minted ones.
template <typename Index, typename Real>
auto filled_mesh(const mesh<Index, Real, 3> &value,
                 const hole_fill_report<Index, Real> &report)
    -> tf::polygons_buffer<Index, Real, 3, 3>;

#define TF_CPP_EXTERN_FILLED_MESH(Index, Real)                                 \
  extern template auto filled_mesh<Index, Real>(                               \
      const mesh<Index, Real, 3> &, const hole_fill_report<Index, Real> &)     \
      -> tf::polygons_buffer<Index, Real, 3, 3>

#if TF_CPP_MATRIX_HAS_3D
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_FILLED_MESH)
#endif

#undef TF_CPP_EXTERN_FILLED_MESH

} // namespace tf::cpp
