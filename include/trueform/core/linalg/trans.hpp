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
#include <array>
namespace tf::linalg {
template <typename T, std::size_t Dims> struct trans {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
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
