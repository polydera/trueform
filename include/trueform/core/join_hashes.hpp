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
namespace tf::core {
namespace detail {
template <typename T> auto join_hashes(T &&t) -> std::size_t { return t; }

template <typename U, typename T, typename... Ts>
auto join_hashes(U out, T &&t, Ts &&...ts) -> std::size_t {
  out ^= t + 0x9e3779b9 + (out << 6) + (out >> 2);
  return join_hashes(out, static_cast<Ts &&>(ts)...);
}
} // namespace detail
template <typename T, typename... Ts>
auto join_hashes(T &&t, Ts &&...ts) -> std::size_t {
  return detail::join_hashes(t, ts...);
}
} // namespace tf::core
