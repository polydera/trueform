/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <type_traits>
namespace tf::linalg {
template <typename T, std::size_t Dims> struct identity {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims + 1;

  auto operator()(std::size_t i, std::size_t j) const -> T { return i == j; }

  constexpr auto rows() const -> std::size_t { return Dims; }

  constexpr auto columns() const -> std::size_t { return Dims + 1; }
};

} // namespace tf::linalg
