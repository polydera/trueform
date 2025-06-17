/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once

#include <array>

namespace tf {
template <typename T, std::size_t Size> struct owned_data {
  using element_type = T;
  using value_type = T;

  owned_data() = default;
  owned_data(std::array<T, Size> _data) : _data{_data} {}

  template <typename Point> auto operator=(const Point &other) -> owned_data & {
    for (std::size_t i = 0; i < Size; ++i)
      _data[i] = other[i];
    return *this;
  }

  auto data() -> T * { return _data.data(); }
  auto data() const -> const T * { return _data.data(); }

private:
  std::array<T, Size> _data;
};

} // namespace tf
