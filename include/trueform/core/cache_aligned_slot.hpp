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
#include <utility>
namespace tf::core {
template <typename T> struct alignas(128) cache_aligned_slot {
  T value;

  cache_aligned_slot() = default;
  cache_aligned_slot(const T &t) : value{t} {}
  cache_aligned_slot(T &&t) : value{std::move(t)} {}

  static constexpr std::size_t alignment = 128;

  static constexpr std::size_t total_size =
      ((sizeof(T) + alignment - 1) / alignment) * alignment;

  static constexpr std::size_t pad_size = total_size - sizeof(T);

  char padding[pad_size > 0 ? pad_size : 0];
};
} // namespace tf::core
