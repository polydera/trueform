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

#include "trueform/cpp/io/read_stl.hpp"

#include "trueform/io/read_stl.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tf::cpp {

template <typename Index>
auto read_stl(std::string_view path)
    -> detail::minted_mesh_result_t<Index, float, 3> {
  return tf::read_stl<Index>(path);
}

template <typename Index>
auto read_stl(io_bytes bytes) -> detail::minted_mesh_result_t<Index, float, 3> {
  return tf::read_stl<Index>(bytes.make_range());
}

template <typename Index>
auto read_stl(const std::int8_t *data, std::size_t size)
    -> detail::minted_mesh_result_t<Index, float, 3> {
  return cpp::read_stl<Index>(io_bytes::copy(data, size));
}

} // namespace tf::cpp
