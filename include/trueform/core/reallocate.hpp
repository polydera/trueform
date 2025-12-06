/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./blocked_buffer.hpp"
#include "./buffer.hpp"
#include "./small_vector.hpp"
#include <vector>

namespace tf::core {

/// @brief Reallocates buffer to size `n`, preserving content (buffer).
template <typename T> auto reallocate(buffer<T> &b, std::size_t n) {
  b.reallocate(n);
}

/// @brief Reallocates to size `n` blocks, preserving content (blocked_buffer).
template <typename T, std::size_t N>
auto reallocate(blocked_buffer<T, N> &b, std::size_t n) {
  b.reallocate(n);
}

/// @brief Resizes vector to `n` elements (std::vector).
template <typename T> auto reallocate(std::vector<T> &v, std::size_t n) {
  v.resize(n);
}

/// @brief Resizes small_vector to `n` elements.
template <typename T, unsigned N>
auto reallocate(small_vector<T, N> &v, std::size_t n) {
  v.resize(n);
}

} // namespace tf::core
