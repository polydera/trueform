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
#include <vector>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Ngon>
struct mesh_arrangement_result {
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> tag_labels;
  nd_array<Index> face_labels;
};

/// @brief Typed arrangement and exact-curve result.
template <typename Index, typename Real, std::size_t Ngon>
struct mesh_arrangement_with_curves_result {
  tf::polygons_buffer<Index, Real, 3, Ngon> mesh;
  nd_array<Index> tag_labels;
  nd_array<Index> face_labels;
  tf::curves_buffer<Index, Real, 3> curves;
};

/// @brief Split and merge the domains of a range of 3D meshes.
///
/// All operands use one index dtype, and an uncut face is emitted verbatim,
/// so the result states the arity the operands did. A range is homogeneous, so
/// its element is what states that arity.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto mesh_arrangements(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                       tf::arrangement_config config = {})
    -> mesh_arrangement_result<Index, Real, Ngon>;

/// @brief The same arrangement, with the exact curves beside it.
template <typename Index, typename Real, std::size_t Ngon>
auto mesh_arrangements_with_curves(
    const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
    tf::arrangement_config config = {})
    -> mesh_arrangement_with_curves_result<Index, Real, Ngon>;

#define TF_CPP_EXTERN_MESH_ARRANGEMENTS(Index, Real, Ngon)                     \
  extern template auto mesh_arrangements<Index, Real, Ngon>(                   \
      const std::vector<mesh<Index, Real, 3, Ngon>> &, tf::arrangement_config) \
      -> mesh_arrangement_result<Index, Real, Ngon>;                           \
  extern template auto mesh_arrangements_with_curves<Index, Real, Ngon>(       \
      const std::vector<mesh<Index, Real, 3, Ngon>> &, tf::arrangement_config) \
      -> mesh_arrangement_with_curves_result<Index, Real, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_MESH_ARRANGEMENTS)

#undef TF_CPP_EXTERN_MESH_ARRANGEMENTS

} // namespace tf::cpp
