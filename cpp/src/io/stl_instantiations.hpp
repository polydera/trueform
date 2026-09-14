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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

#define TF_CPP_INSTANTIATE_READ_STL(Index)                                     \
  template auto read_stl<Index>(std::string_view)                              \
      ->detail::minted_mesh_result_t<Index, float, 3>;                           \
  template auto read_stl<Index>(io_bytes)                                      \
      ->detail::minted_mesh_result_t<Index, float, 3>;                           \
  template auto read_stl<Index>(const std::int8_t *, std::size_t)              \
      -> detail::minted_mesh_result_t<Index, float, 3>

#define TF_CPP_INSTANTIATE_WRITE_STL(Index, Real)                              \
  template auto write_stl<Index, Real, 3, 3>(const mesh<Index, Real, 3, 3> &)  \
      -> nd_array<std::int8_t>;                                                \
  template auto write_stl<Index, Real, 3, 3>(                                  \
      const mesh<Index, Real, 3, 3> &, const std::filesystem::path &) -> bool
