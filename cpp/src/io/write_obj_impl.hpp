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

#include "trueform/cpp/io/write_obj.hpp"

#include "trueform/core/policy/frame.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/io/write_obj.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <utility>

namespace tf::cpp {
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto write_obj(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<std::int8_t> {
  value.require_indices();
  auto bytes = tf::write_obj_to_buffer<std::int8_t>(value.polygons() |
                                                    tf::tag(value.frame()));
  const auto size = static_cast<int>(bytes.size());
  return nd_array<std::int8_t>::from_buffer(std::move(bytes), {size});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto write_obj(const mesh<Index, Real, Dims, Ngon> &value,
               const std::filesystem::path &path) -> bool {
  value.require_indices();
  return tf::write_obj(value.polygons() | tf::tag(value.frame()),
                       path.string());
}

} // namespace tf::cpp
