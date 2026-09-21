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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Whether a 3D mesh meets itself, exactly and at the first contact.
/// @note The operand is read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the tree, the face membership and
/// the manifold edge link — and a cache already holding them is what a
/// second ask costs nothing against.
template <typename Index, typename Real, std::size_t Ngon>
auto has_self_intersections(const mesh<Index, Real, 3, Ngon> &value) -> bool;

#define TF_CPP_EXTERN_HAS_SELF_INTERSECTIONS(Index, Real, Ngon)                \
  extern template auto has_self_intersections<Index, Real, Ngon>(              \
      const mesh<Index, Real, 3, Ngon> &) -> bool

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_HAS_SELF_INTERSECTIONS)

#undef TF_CPP_EXTERN_HAS_SELF_INTERSECTIONS

} // namespace tf::cpp
