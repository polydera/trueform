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
#include "../core/point.hpp"
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
