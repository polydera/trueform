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
#include <type_traits>
namespace tf::linalg {
template <typename T, std::size_t Dims> struct identity {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims + 1;

  auto operator()(std::size_t i, std::size_t j) const -> T { return i == j; }

  constexpr auto rows() const -> std::size_t { return Dims; }

  constexpr auto columns() const -> std::size_t { return Dims + 1; }
};

} // namespace tf::linalg
