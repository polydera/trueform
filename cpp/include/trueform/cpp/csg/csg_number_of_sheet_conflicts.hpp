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
#include "trueform/cpp/csg/csg_graph.hpp"

namespace tf::cpp {

/// @brief How many `(sheet, domain)` pairs two volumes the sheet separates
/// state opposite sides of; such a domain reads no side of the sheet.
template <typename Index, typename Real>
auto csg_number_of_sheet_conflicts(const csg_graph<Index, Real> &graph)
    -> Index;

#define TF_CPP_EXTERN_CSG_NUMBER_OF_SHEET_CONFLICTS(Index, Real)               \
  extern template auto csg_number_of_sheet_conflicts(                          \
      const csg_graph<Index, Real> &) -> Index

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_EXTERN_CSG_NUMBER_OF_SHEET_CONFLICTS)

#undef TF_CPP_EXTERN_CSG_NUMBER_OF_SHEET_CONFLICTS

} // namespace tf::cpp
