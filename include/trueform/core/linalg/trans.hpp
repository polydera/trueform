/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <array>
namespace tf::linalg {
template <typename T, std::size_t Dims> struct trans {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims + 1;

  trans() = default;
  trans(std::array<std::array<T, Dims + 1>, Dims> _trans) : _trans{_trans} {}

  trans(std::initializer_list<T> list) {
    auto _ptr = list.begin();
    for (std::size_t i = 0; i < n_rows; ++i) {
      for (std::size_t j = 0; j < n_columns; ++j)
        (*this)(i, j) = *_ptr++;
    }
  }

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return _trans[i][j];
  }

  auto operator()(std::size_t i, std::size_t j) -> decltype(auto) {
    return _trans[i][j];
  }

private:
  std::array<std::array<T, Dims + 1>, Dims> _trans;
};
} // namespace tf::linalg
