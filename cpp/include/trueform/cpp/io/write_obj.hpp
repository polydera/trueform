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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>

namespace tf::cpp {

/// @brief Serialize a 3D mesh to ASCII OBJ bytes, at either arity.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<std::int8_t>;

/// @brief Write a 3D mesh to an OBJ filesystem path.
/// @param path Output path. A lowercase .obj suffix is appended when absent.
/// @return true after a complete write, false when serialization is empty or
/// the file cannot be opened or written.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(const mesh<Index, Real, Dims, Ngon> &value,
               const std::filesystem::path &path) -> bool;

#define TF_CPP_EXTERN_WRITE_OBJ(Index, Real, Ngon)                             \
  extern template auto write_obj<Index, Real, 3, Ngon>(                        \
      const mesh<Index, Real, 3, Ngon> &) -> nd_array<std::int8_t>;            \
  extern template auto write_obj<Index, Real, 3, Ngon>(                        \
      const mesh<Index, Real, 3, Ngon> &, const std::filesystem::path &)       \
      -> bool

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_WRITE_OBJ)

#undef TF_CPP_EXTERN_WRITE_OBJ

} // namespace tf::cpp
