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

#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/io/bytes.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tf::cpp {

/// @brief Read a binary or ASCII STL file into core's own storage.
///
/// STL is float triangles by the format's own definition, so the result states
/// that arity and that real. It is the caller's to hold; assemble a `mesh` over
/// it and a `cache` to ask anything of it.
template <typename Index = default_index_t>
auto read_stl(std::string_view path)
    -> detail::minted_mesh_result_t<Index, float, 3>;

/// @brief Read binary or ASCII STL bytes from an owning parser-ready payload.
template <typename Index = default_index_t>
auto read_stl(io_bytes bytes) -> detail::minted_mesh_result_t<Index, float, 3>;

/// @brief Read binary or ASCII STL bytes from a non-owning byte span.
template <typename Index = default_index_t>
auto read_stl(const std::int8_t *data, std::size_t size)
    -> detail::minted_mesh_result_t<Index, float, 3>;

#define TF_CPP_EXTERN_READ_STL(Index)                                          \
  extern template auto read_stl<Index>(std::string_view)                       \
      ->detail::minted_mesh_result_t<Index, float, 3>;                         \
  extern template auto read_stl<Index>(io_bytes)                               \
      ->detail::minted_mesh_result_t<Index, float, 3>;                         \
  extern template auto read_stl<Index>(const std::int8_t *, std::size_t)       \
      -> detail::minted_mesh_result_t<Index, float, 3>

#if TF_CPP_MATRIX_HAS_FLOAT
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_READ_STL)
#endif

#undef TF_CPP_EXTERN_READ_STL

} // namespace tf::cpp
