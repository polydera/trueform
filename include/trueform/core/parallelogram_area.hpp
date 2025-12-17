/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./dot.hpp"
#include "./sqrt.hpp"
#include "./vector_like.hpp"
namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto parallelogram_area2(const vector_like<Dims, Policy0> &v0,
                         const vector_like<Dims, Policy1> &v1) {
  auto dot = tf::dot(v0, v1);
  return v0.length2() * v1.length2() - dot * dot;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto parallelogram_area(const vector_like<Dims, Policy0> &v0,
                        const vector_like<Dims, Policy1> &v1) {
  return tf::sqrt(parallelogram_area(v0, v1));
}

template <typename Policy0, typename Policy1>
auto parallelogram_area2(const vector_like<2, Policy0> &v0,
                         const vector_like<2, Policy1> &v1) {
  auto tmp = v0[0] * v1[1] - v0[1] * v1[0];
  return tmp * tmp;
}

template <typename Policy0, typename Policy1>
auto parallelogram_area(const vector_like<2, Policy0> &v0,
                        const vector_like<2, Policy1> &v1) {
  auto tmp = v0[0] * v1[1] - v0[1] * v1[0];
  return std::abs(tmp);
}

template <typename Policy0, typename Policy1>
auto signed_parallelogram_area(const vector_like<2, Policy0> &v0,
                               const vector_like<2, Policy1> &v1) {
  return v0[0] * v1[1] - v0[1] * v1[0];
}
} // namespace tf
