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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/detail/supported_outer_shell.hpp"
#include "trueform/intersect/intersect_config.hpp"
#include "trueform/intersect/intersect_mode.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Repair a 3D mesh to its outer shell, at either arity.
///
/// The shell is the operand's own boundary: an uncut face is emitted verbatim,
/// so the result states the arity the operand did. Connectivity retains the
/// operand's index dtype. Geometry is read in the raw local frame, so a stated
/// placement is not applied.
/// @note The operands are read through `mesh::topology_form()`, so what a
/// caller prebuilds for this family is the TREE, the FACE MEMBERSHIP and
/// the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Ngon>
auto outer_shell(const mesh<Index, Real, 3, Ngon> &value,
                 tf::intersect_config intersect_config =
                     {tf::intersect_mode::primitives |
                      tf::intersect_mode::resolve_contours})
    -> tf::polygons_buffer<Index, Real, 3, Ngon>;

// An unsupported index or real has no compiled entry, so it is refused where
// it is written rather than at the link.
template <
    typename Index, typename Real, std::size_t Ngon,
    std::enable_if_t<!detail::is_supported_outer_shell_v<Index, Real>, int> = 0>
auto outer_shell(const mesh<Index, Real, 3, Ngon> &,
                 tf::intersect_config = {tf::intersect_mode::primitives |
                                         tf::intersect_mode::resolve_contours})
    -> void = delete;

#define TF_CPP_EXTERN_TYPED_OUTER_SHELL(Index, Real, Ngon)                     \
  extern template auto outer_shell<Index, Real, Ngon>(                         \
      const mesh<Index, Real, 3, Ngon> &, tf::intersect_config)                \
      -> tf::polygons_buffer<Index, Real, 3, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_TYPED_OUTER_SHELL)

#undef TF_CPP_EXTERN_TYPED_OUTER_SHELL

} // namespace tf::cpp
