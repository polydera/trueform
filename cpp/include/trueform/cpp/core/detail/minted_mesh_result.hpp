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
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp::detail {

/// What an entry that MINTS a mesh hands back — a read, a primitive, anything
/// standing on no operand: core's own storage, at the arity it was asked for.
/// A combination this build left out has no `type`, so a caller naming it is
/// refused where it is written rather than at the link.
template <typename Index, typename Real, std::size_t Ngon, typename = void>
struct minted_mesh_result {};

template <typename Index, typename Real, std::size_t Ngon>
struct minted_mesh_result<
    Index, Real, Ngon,
    std::enable_if_t<is_supported_index_v<Index> &&
                     std::is_same_v<Index, std::remove_cv_t<Index>> &&
                     matrix_has_index_v<Index> &&
                     (std::is_same_v<Real, float> ||
                      std::is_same_v<Real, double>) &&
                     matrix_carries_real_v<Real> && matrix_has_ngon_v<Ngon>>> {
  using type = tf::polygons_buffer<Index, Real, 3, Ngon>;
};

template <typename Index, typename Real, std::size_t Ngon>
using minted_mesh_result_t =
    typename minted_mesh_result<Index, Real, Ngon>::type;

} // namespace tf::cpp::detail
