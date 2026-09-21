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

#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/core/curves_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Typed self-arrangement result retaining source-face identities.
template <typename Index, typename Real, std::size_t Ngon>
struct polygon_arrangement_result {
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> face_labels;
};

/// @brief Typed self-arrangement and exact-curve result.
template <typename Index, typename Real, std::size_t Ngon>
struct polygon_arrangement_with_curves_result {
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> face_labels;
  tf::curves_buffer<Index, Real, 3> curves;
};

/// @brief Split a 3D mesh at its self intersections, at either arity.
/// The result carries the operand's own layout. Stored transformations are
/// intentionally ignored for this local-frame operation.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto polygon_arrangements(const mesh<Index, Real, 3, Ngon> &value,
                          tf::arrangement_config config = {})
    -> polygon_arrangement_result<Index, Real, Ngon>;

/// @brief The same self arrangement, with the exact curves beside it.
template <typename Index, typename Real, std::size_t Ngon>
auto polygon_arrangements_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                                      tf::arrangement_config config = {})
    -> polygon_arrangement_with_curves_result<Index, Real, Ngon>;

#define TF_CPP_EXTERN_POLYGON_ARRANGEMENTS(Index, Real, Ngon)                  \
  extern template auto polygon_arrangements<Index, Real, Ngon>(                \
      const mesh<Index, Real, 3, Ngon> &, tf::arrangement_config)              \
      -> polygon_arrangement_result<Index, Real, Ngon>;                        \
  extern template auto polygon_arrangements_with_curves<Index, Real, Ngon>(    \
      const mesh<Index, Real, 3, Ngon> &, tf::arrangement_config)              \
      -> polygon_arrangement_with_curves_result<Index, Real, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_POLYGON_ARRANGEMENTS)

#undef TF_CPP_EXTERN_POLYGON_ARRANGEMENTS

} // namespace tf::cpp
