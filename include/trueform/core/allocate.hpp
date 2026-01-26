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
#include "./blocked_buffer.hpp"
#include "./buffer.hpp"
#include "./small_vector.hpp"
#include <vector>

namespace tf::core {

/// @brief Allocates uninitialized memory for `n` elements (buffer).
template <typename T> auto allocate(buffer<T> &b, std::size_t n) {
  b.allocate(n);
}

/// @brief Allocates uninitialized memory for `n` blocks (blocked_buffer).
template <typename T, std::size_t N>
auto allocate(blocked_buffer<T, N> &b, std::size_t n) {
  b.allocate(n);
}

/// @brief Resizes vector to `n` elements (std::vector).
template <typename T> auto allocate(std::vector<T> &v, std::size_t n) {
  v.resize(n);
}

/// @brief Resizes small_vector to `n` elements.
template <typename T, unsigned N>
auto allocate(small_vector<T, N> &v, std::size_t n) {
  v.resize(n);
}

} // namespace tf::core
