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

#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/io/bytes.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tf::cpp {

/// @brief Read an OBJ into core's own storage, at the arity asked for.
///
/// An OBJ states its own faces and a file may hold any of them, so the arity a
/// caller reads at is the request: the default is the mixed one, and
/// `read_obj<Index, Real, 3>` states triangles and reads a file holding
/// anything else as nothing. The result is the caller's to hold; assemble a
/// `mesh` over it and a `cache` to ask anything of it.
template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(std::string_view path)
    -> detail::minted_mesh_result_t<Index, Real, Ngon>;

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(io_bytes bytes)
    -> detail::minted_mesh_result_t<Index, Real, Ngon>;

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(const std::int8_t *data, std::size_t size)
    -> detail::minted_mesh_result_t<Index, Real, Ngon>;

#define TF_CPP_EXTERN_READ_OBJ(Index, Real, Ngon)                              \
  extern template auto read_obj<Index, Real, Ngon>(std::string_view)           \
      ->detail::minted_mesh_result_t<Index, Real, Ngon>;                       \
  extern template auto read_obj<Index, Real, Ngon>(io_bytes)                   \
      ->detail::minted_mesh_result_t<Index, Real, Ngon>;                       \
  extern template auto read_obj<Index, Real, Ngon>(const std::int8_t *,        \
                                                   std::size_t)                \
      -> detail::minted_mesh_result_t<Index, Real, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_READ_OBJ)

#undef TF_CPP_EXTERN_READ_OBJ

} // namespace tf::cpp
