/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./point.hpp"
#include "./random_vector.hpp"

namespace tf {
template <int N, typename T>
auto random_point(T from, T to) -> tf::point<T, N> {
  return tf::make_point(tf::random_vector(from, to));
}

template <typename T, std::size_t N> auto random_point() -> tf::point<T, N> {
  return tf::make_point(tf::random_vector<T, N>());
}
} // namespace tf
