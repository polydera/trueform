/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/memory.hpp"
#include "../../core/range.hpp"
#include <cstddef>
#include <memory>

namespace tf::geometry {

/// One record of atomics per point, written by concurrent workers.
/// `tf::allocate` returns raw storage, not live atomics, so every record is
/// placement-constructed from the same arguments when the storage is sized.
template <typename T> class atomic_values {
public:
  template <typename... Args>
  auto allocate_and_fill(std::size_t n, const Args &...args) -> void {
    _values.reset(tf::allocate<T>(n));
    _size = n;
    tf::parallel_for_each(
        values(),
        [&](T &slot) { ::new (static_cast<void *>(&slot)) T(args...); },
        tf::checked);
  }

  auto operator[](std::size_t i) -> T & { return _values[i]; }

  auto values() -> tf::range<T *, tf::dynamic_size> {
    return tf::make_range(_values.get(), _size);
  }

  auto size() const -> std::size_t { return _size; }

private:
  struct values_deleter {
    auto operator()(void *p) const noexcept -> void { tf::deallocate<T>(p); }
  };

  std::unique_ptr<T[], values_deleter> _values;
  std::size_t _size = 0;
};

} // namespace tf::geometry
