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
#include "./zip_apply.hpp"
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

template <typename T> auto append(const buffer<T> &input, buffer<T> &output) {
  auto old_data_size = output.size();
  reallocate(output, old_data_size + input.size());
  std::copy(input.begin(), input.end(), output.begin() + old_data_size);
}

template <typename T, std::size_t N>
auto append(const blocked_buffer<T, N> &input, blocked_buffer<T, N> &output) {
  auto old_data_size = output.size();
  reallocate(output, old_data_size + input.size());
  std::copy(input.begin(), input.end(), output.begin() + old_data_size);
}

template <typename T>
auto append(const std::vector<T> &input, std::vector<T> &output) {
  auto old_data_size = output.size();
  reallocate(output, old_data_size + input.size());
  std::copy(input.begin(), input.end(), output.begin() + old_data_size);
}

template <typename T, unsigned N>
auto append(const small_vector<T, N> &input, small_vector<T, N> &output) {
  auto old_data_size = output.size();
  reallocate(output, old_data_size + input.size());
  std::copy(input.begin(), input.end(), output.begin() + old_data_size);
}

template <typename... Buffers0, typename... Buffers1>
auto append(const std::tuple<Buffers0...> &input,
            std::tuple<Buffers1...> &output) {
  tf::zip_apply(
      [](auto &&...pairs) {
        using std::get;
        (append(get<0>(pairs), get<1>(pairs)), ...);
      },
      input, output);
}

template <typename T> auto reserve(buffer<T> &buff, std::size_t n) {
  buff.reserve(n);
}

template <typename T, std::size_t N>
auto reserve(blocked_buffer<T, N> &buff, std::size_t n) {
  buff.reserve(n);
}

template <typename T> auto reserve(std::vector<T> &buff, std::size_t n) {
  buff.reserve(n);
}

template <typename T, unsigned N>
auto reserve(small_vector<T, N> &buff, std::size_t n) {
  buff.reserve(n);
}

template <typename... Buffers>
auto reserve(std::tuple<Buffers...> &buff, std::size_t n) {
  tf::apply([&](auto &&...elem) { (elem.reserve(n), ...); }, buff);
}

template <typename T> auto size(const buffer<T> &buff) { return buff.size(); }

template <typename T, std::size_t N>
auto size(const blocked_buffer<T, N> &buff) {
  return buff.size();
}

template <typename T> auto size(const std::vector<T> &buff) {
  return buff.size();
}

template <typename T, unsigned N> auto size(const small_vector<T, N> &buff) {
  return buff.size();
}

template <typename... Buffers> auto size(const std::tuple<Buffers...> &buff) {
  using std::get;
  return get<0>(buff).size();
}

} // namespace tf::core
