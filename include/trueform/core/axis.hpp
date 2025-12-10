/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf {

template <std::size_t I> struct axis_t {
  template <typename T, std::size_t Dims>
  constexpr operator unit_vector<T, Dims>() const {
    static_assert(I < Dims, "Axis index must be less than dimensions");
    tf::vector<T, Dims> data{};
    data[I] = T{1};
    return unit_vector<T, Dims>{tf::unsafe, data};
  }
};

template <std::size_t I> inline constexpr axis_t<I> axis{};

template <typename T, std::size_t Dims, std::size_t I>
auto make_unit_vector(tf::axis_t<I> a) -> tf::unit_vector<T, Dims> {
  return a;
}

} // namespace tf
