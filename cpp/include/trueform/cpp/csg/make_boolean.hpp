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
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/detail/concatenated_arity.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/csg/boolean_op.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tf::cpp {

/// @brief Typed boolean result preserving the lossless common index width.
template <typename Index, typename Real, std::size_t Ngon>
struct boolean_result {
  static_assert(is_supported_index_v<Index>,
                "boolean_result requires a supported index type");
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<std::int8_t> labels;
  nd_array<Index> face_labels;
};

/// @brief Typed boolean result with exact curves in the common index width.
template <typename Index, typename Real, std::size_t Ngon>
struct boolean_with_curves_result {
  static_assert(is_supported_index_v<Index>,
                "boolean_with_curves_result requires a supported index type");
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<std::int8_t> labels;
  nd_array<Index> face_labels;
  tf::curves_buffer<Index, Real, 3> curves;
};

/// @brief Perform a boolean on two 3D meshes at either arity.
/// Mixed index operands produce mesh connectivity and face labels in their
/// lossless common index type.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto make_boolean(const mesh<Index0, Real, 3, Ngon0> &a,
                  const mesh<Index1, Real, 3, Ngon1> &b,
                  tf::boolean_op operation,
                  std::vector<std::int32_t> sheets = {},
                  tf::arrangement_config config = {})
    -> boolean_result<common_index_t<Index0, Index1>, Real,
                      detail::concatenated_arity_v<Ngon0, Ngon1>>;

/// @brief The same boolean, with the exact intersection curves beside it.
template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto make_boolean_with_curves(const mesh<Index0, Real, 3, Ngon0> &a,
                              const mesh<Index1, Real, 3, Ngon1> &b,
                              tf::boolean_op operation,
                              std::vector<std::int32_t> sheets = {},
                              tf::arrangement_config config = {})
    -> boolean_with_curves_result<common_index_t<Index0, Index1>, Real,
                                  detail::concatenated_arity_v<Ngon0, Ngon1>>;

#define TF_CPP_BOOLEAN_TYPED_EXTERN(Index0, Real, Index1, Ngon0, Ngon1)        \
  extern template auto make_boolean<Index0, Real, Index1, Ngon0, Ngon1>(       \
      const mesh<Index0, Real, 3, Ngon0> &,                                    \
      const mesh<Index1, Real, 3, Ngon1> &, tf::boolean_op,                    \
      std::vector<std::int32_t>, tf::arrangement_config)                       \
      -> boolean_result<common_index_t<Index0, Index1>, Real,                  \
                        detail::concatenated_arity_v<Ngon0, Ngon1>>;           \
  extern template auto                                                         \
  make_boolean_with_curves<Index0, Real, Index1, Ngon0, Ngon1>(                \
      const mesh<Index0, Real, 3, Ngon0> &,                                    \
      const mesh<Index1, Real, 3, Ngon1> &, tf::boolean_op,                    \
      std::vector<std::int32_t>, tf::arrangement_config)                       \
      -> boolean_with_curves_result<                                           \
          common_index_t<Index0, Index1>, Real,                                \
          detail::concatenated_arity_v<Ngon0, Ngon1>>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_INDEX_NGON_PAIR(TF_CPP_BOOLEAN_TYPED_EXTERN)

#undef TF_CPP_BOOLEAN_TYPED_EXTERN

} // namespace tf::cpp
