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
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::cpp {

/// @brief Typed-index isobands result preserving the input mesh carrier.
template <typename Index, typename Real, std::size_t Ngon>
struct isobands_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "isobands_result requires an unqualified supported "
                "index type");

  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> labels;
  nd_array<Index> face_labels;
};

/// @brief Typed-index isobands and contour-curves result.
template <typename Index, typename Real, std::size_t Ngon>
struct isobands_with_curves_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "isobands_with_curves_result requires an unqualified "
                "supported index type");

  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> labels;
  nd_array<Index> face_labels;
  tf::curves_buffer<Index, Real, 3> curves;
};

/// @brief Every band of a 3D mesh's scalar field, at either arity.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto isobands(const mesh<Index, Real, 3, Ngon> &value,
              const nd_array<Real> &scalars, const nd_array<Real> &cut_values)
    -> isobands_result<Index, Real, Ngon>;

/// @brief The same bands, with the contour curves beside them.
template <typename Index, typename Real, std::size_t Ngon>
auto isobands_with_curves(const mesh<Index, Real, 3, Ngon> &value,
                          const nd_array<Real> &scalars,
                          const nd_array<Real> &cut_values)
    -> isobands_with_curves_result<Index, Real, Ngon>;

/// @brief The named bands alone.
template <typename Index, typename Real, std::size_t Ngon>
auto isobands_selected(const mesh<Index, Real, 3, Ngon> &value,
                       const nd_array<Real> &scalars,
                       const nd_array<Real> &cut_values,
                       const nd_array<std::int32_t> &selected_bands)
    -> isobands_result<Index, Real, Ngon>;

/// @brief The named bands alone, with their contour curves.
template <typename Index, typename Real, std::size_t Ngon>
auto isobands_with_curves_selected(const mesh<Index, Real, 3, Ngon> &value,
                                   const nd_array<Real> &scalars,
                                   const nd_array<Real> &cut_values,
                                   const nd_array<std::int32_t> &selected_bands)
    -> isobands_with_curves_result<Index, Real, Ngon>;

#define TF_CPP_ISOBANDS_TYPED_EXTERN(Index, Real, Ngon)                        \
  extern template auto isobands<Index, Real, Ngon>(                            \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &,              \
      const nd_array<Real> &) -> isobands_result<Index, Real, Ngon>;           \
  extern template auto isobands_with_curves<Index, Real, Ngon>(                \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &,              \
      const nd_array<Real> &)                                                  \
      -> isobands_with_curves_result<Index, Real, Ngon>;                       \
  extern template auto isobands_selected<Index, Real, Ngon>(                   \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &,              \
      const nd_array<Real> &, const nd_array<std::int32_t> &)                  \
      -> isobands_result<Index, Real, Ngon>;                                   \
  extern template auto isobands_with_curves_selected<Index, Real, Ngon>(       \
      const mesh<Index, Real, 3, Ngon> &, const nd_array<Real> &,              \
      const nd_array<Real> &, const nd_array<std::int32_t> &)                  \
      -> isobands_with_curves_result<Index, Real, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_ISOBANDS_TYPED_EXTERN)

#undef TF_CPP_ISOBANDS_TYPED_EXTERN

} // namespace tf::cpp
