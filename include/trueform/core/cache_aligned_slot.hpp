/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <cstddef> // std::size_t
namespace tf::core {
template <typename T> struct alignas(128) cache_aligned_slot {
  T value;

  static constexpr std::size_t alignment = 128;

  static constexpr std::size_t total_size =
      ((sizeof(T) + alignment - 1) / alignment) * alignment;

  static constexpr std::size_t pad_size = total_size - sizeof(T);

  char padding[pad_size > 0 ? pad_size : 0];
};
} // namespace tf::core
