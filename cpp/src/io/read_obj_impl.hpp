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

#include "trueform/cpp/io/read_obj.hpp"

#include "trueform/io/read_obj.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Ngon>
auto read_obj(std::string_view path)
    -> detail::minted_mesh_result_t<Index, Real, Ngon> {
  return tf::read_obj<Index, Ngon, Real>(path);
}

template <typename Index, typename Real, std::size_t Ngon>
auto read_obj(io_bytes bytes) -> detail::minted_mesh_result_t<Index, Real, Ngon> {
  return tf::read_obj<Index, Ngon, Real>(bytes.make_range());
}

template <typename Index, typename Real, std::size_t Ngon>
auto read_obj(const std::int8_t *data, std::size_t size)
    -> detail::minted_mesh_result_t<Index, Real, Ngon> {
  return cpp::read_obj<Index, Real, Ngon>(io_bytes::copy(data, size));
}

} // namespace tf::cpp
