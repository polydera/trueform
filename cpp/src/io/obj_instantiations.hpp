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

#define TF_CPP_INSTANTIATE_OBJ_IO(Index, Real, Ngon)                           \
  template auto read_obj<Index, Real, Ngon>(std::string_view)                  \
      ->detail::minted_mesh_result_t<Index, Real, Ngon>;                         \
  template auto read_obj<Index, Real, Ngon>(io_bytes)                          \
      ->detail::minted_mesh_result_t<Index, Real, Ngon>;                         \
  template auto read_obj<Index, Real, Ngon>(const std::int8_t *, std::size_t)  \
      -> detail::minted_mesh_result_t<Index, Real, Ngon>;                        \
  template auto write_obj<Index, Real, 3, Ngon>(                               \
      const mesh<Index, Real, 3, Ngon> &) -> nd_array<std::int8_t>;            \
  template auto write_obj<Index, Real, 3, Ngon>(                               \
      const mesh<Index, Real, 3, Ngon> &, const std::filesystem::path &)       \
      -> bool
