/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./dot.hpp"
#include "./line_like.hpp"
#include "./polygon.hpp"

namespace tf {
template <typename T> struct interval {
  T min;
  T max;
};

namespace core {
template <typename Range, std::size_t Dims, typename Policy>
auto make_interval(const Range &r, const tf::line_like<Dims, Policy> &line) {
  std::decay_t<decltype(r[0][0])> low, high;
  low = high = tf::dot(r[0] - line.origin, line.direction);
  auto size = r.size();
  for (decltype(size) i = 1; i < size; i++) {
    decltype(low) tmp = tf::dot(r[i] - line.origin, line.direction);
    low = std::min(low, tmp);
    high = std::max(high, tmp);
  }
  return interval<decltype(low)>{low, high};
}
} // namespace core
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_interval(const polygon<Dims, Policy0> &poly,
                   const tf::line_like<Dims, Policy1> &line) {
  return core::make_interval(poly, line);
}
} // namespace tf
