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

#include "trueform/core/buffer.hpp"
#include "trueform/core/range.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tf::cpp {

/// @brief Owning parser-ready IO bytes with a trailing sentinel.
class io_bytes {
  tf::buffer<std::int8_t> _storage;
  std::size_t _size;
  std::int8_t _empty_sentinel = 0;

  explicit io_bytes(std::size_t size) : _size(size) {
    if (size == std::numeric_limits<std::size_t>::max())
      throw std::length_error("IO byte payload size overflows sentinel storage");
    _storage.allocate(size + 1);
    _storage[size] = 0;
  }

  auto reset_empty() noexcept -> void {
    _storage = {};
    _size = 0;
    _empty_sentinel = 0;
  }

public:
  io_bytes(const io_bytes &) = delete;
  auto operator=(const io_bytes &) -> io_bytes & = delete;

  io_bytes(io_bytes &&other) noexcept
      : _storage(std::move(other._storage)), _size(other._size) {
    other.reset_empty();
  }
  auto operator=(io_bytes &&other) noexcept -> io_bytes & {
    if (this == &other)
      return *this;
    _storage = std::move(other._storage);
    _size = other._size;
    _empty_sentinel = 0;
    other.reset_empty();
    return *this;
  }

  static auto allocate(std::size_t size) -> io_bytes { return io_bytes(size); }

  static auto copy(const std::int8_t *data, std::size_t size) -> io_bytes {
    auto bytes = io_bytes(size);
    if (data == nullptr && size != 0)
      throw std::invalid_argument("IO byte data is null");
    if (size != 0)
      std::memcpy(bytes.data(), data, size);
    return bytes;
  }

  auto data() -> std::int8_t * {
    return _storage.data() == nullptr ? &_empty_sentinel : _storage.data();
  }
  auto data() const -> const std::int8_t * {
    return _storage.data() == nullptr ? &_empty_sentinel : _storage.data();
  }
  auto size() const -> std::size_t { return _size; }

  auto make_range() const {
    return tf::make_range(reinterpret_cast<const char *>(data()), _size);
  }
};

} // namespace tf::cpp
