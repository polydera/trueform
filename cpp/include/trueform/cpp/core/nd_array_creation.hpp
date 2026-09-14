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

#include "trueform/core/small_vector.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>

namespace tf::cpp {

template <typename T> auto zeros(tf::small_vector<int, 3> shape) -> nd_array<T>;

template <typename T> auto ones(tf::small_vector<int, 3> shape) -> nd_array<T>;

template <typename T>
auto full(tf::small_vector<int, 3> shape, T value) -> nd_array<T>;

template <typename T> auto eye(int size) -> nd_array<T>;

template <typename T> auto arange(T start, T stop, T step) -> nd_array<T>;

template <typename T> auto linspace(T start, T stop, int count) -> nd_array<T>;

template <typename T>
auto random(tf::small_vector<int, 3> shape, T lower, T upper) -> nd_array<T>;

#define TF_CPP_EXTERN_FILLED(T)                                                \
  extern template auto zeros<T>(tf::small_vector<int, 3>)->nd_array<T>;        \
  extern template auto ones<T>(tf::small_vector<int, 3>)->nd_array<T>;         \
  extern template auto full<T>(tf::small_vector<int, 3>, T) -> nd_array<T>

TF_CPP_EXTERN_FILLED(std::int8_t);
TF_CPP_EXTERN_FILLED(std::int32_t);
TF_CPP_EXTERN_FILLED(float);
TF_CPP_EXTERN_FILLED(double);

#undef TF_CPP_EXTERN_FILLED

#define TF_CPP_EXTERN_RANGE(T)                                                 \
  extern template auto eye<T>(int) -> nd_array<T>;                             \
  extern template auto arange<T>(T, T, T) -> nd_array<T>

TF_CPP_EXTERN_RANGE(std::int32_t);
TF_CPP_EXTERN_RANGE(float);
TF_CPP_EXTERN_RANGE(double);

#undef TF_CPP_EXTERN_RANGE

extern template auto linspace<float>(float, float, int) -> nd_array<float>;
extern template auto linspace<double>(double, double, int) -> nd_array<double>;

#define TF_CPP_EXTERN_RANDOM(T)                                                \
  extern template auto random<T>(tf::small_vector<int, 3>, T, T) -> nd_array<T>

TF_CPP_EXTERN_RANDOM(std::int32_t);
TF_CPP_EXTERN_RANDOM(float);
TF_CPP_EXTERN_RANDOM(double);

#undef TF_CPP_EXTERN_RANDOM

} // namespace tf::cpp
