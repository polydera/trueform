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
#include "trueform/cpp/core/nd_array_creation.hpp"

#include "shape_size.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/small_vector.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/random/random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {

template <typename T>
auto zeros(tf::small_vector<int, 3> shape) -> nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(detail::shape_size(shape));
  std::memset(buffer.data(), 0, buffer.size() * sizeof(T));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T> auto ones(tf::small_vector<int, 3> shape) -> nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(detail::shape_size(shape));
  std::fill(buffer.begin(), buffer.end(), T{1});
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto full(tf::small_vector<int, 3> shape, T value) -> nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(detail::shape_size(shape));
  std::fill(buffer.begin(), buffer.end(), value);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T> auto eye(int size) -> nd_array<T> {
  auto result = zeros<T>({size, size});
  for (int index = 0; index < size; ++index)
    result[static_cast<std::size_t>(index * size + index)] = T{1};
  return result;
}

template <typename T> auto arange(T start, T stop, T step) -> nd_array<T> {
  std::size_t length = 0;
  if constexpr (std::is_floating_point_v<T>) {
    if (step > 0 && stop > start)
      length = static_cast<std::size_t>(std::ceil(
          static_cast<double>(stop - start) / static_cast<double>(step)));
    else if (step < 0 && stop < start)
      length = static_cast<std::size_t>(std::ceil(
          static_cast<double>(start - stop) / static_cast<double>(-step)));
  } else {
    if (step > 0 && stop > start)
      length = static_cast<std::size_t>((stop - start + step - 1) / step);
    else if (step < 0 && stop < start)
      length = static_cast<std::size_t>((start - stop - step - 1) / (-step));
  }

  tf::buffer<T> buffer;
  buffer.allocate(length);
  auto value = start;
  for (std::size_t index = 0; index < length; ++index) {
    buffer[index] = value;
    value += step;
  }
  return nd_array<T>::from_buffer(std::move(buffer),
                                  {static_cast<int>(length)});
}

template <typename T> auto linspace(T start, T stop, int count) -> nd_array<T> {
  if (count < 0)
    throw std::invalid_argument("linspace: count must be nonnegative");
  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(count));
  if (count == 1) {
    buffer[0] = start;
  } else {
    const auto step = (stop - start) / static_cast<T>(count - 1);
    for (int index = 0; index < count; ++index)
      buffer[static_cast<std::size_t>(index)] =
          start + static_cast<T>(index) * step;
  }
  return nd_array<T>::from_buffer(std::move(buffer), {count});
}

template <typename T>
auto random(tf::small_vector<int, 3> shape, T lower, T upper) -> nd_array<T> {
  const auto total = detail::shape_size(shape);
  tf::buffer<T> buffer;
  buffer.allocate(total);
  auto *output = buffer.data();
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(total)),
      [=](int index) { output[index] = tf::random(lower, upper); },
      tf::checked);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

#define TF_CPP_INSTANTIATE_FILLED(T)                                           \
  template auto zeros<T>(tf::small_vector<int, 3>)->nd_array<T>;               \
  template auto ones<T>(tf::small_vector<int, 3>)->nd_array<T>;                \
  template auto full<T>(tf::small_vector<int, 3>, T) -> nd_array<T>

TF_CPP_INSTANTIATE_FILLED(std::int8_t);
TF_CPP_INSTANTIATE_FILLED(std::int32_t);
TF_CPP_INSTANTIATE_FILLED(std::int64_t);
TF_CPP_INSTANTIATE_FILLED(float);
TF_CPP_INSTANTIATE_FILLED(double);

#undef TF_CPP_INSTANTIATE_FILLED

#define TF_CPP_INSTANTIATE_RANGE(T)                                            \
  template auto eye<T>(int) -> nd_array<T>;                                    \
  template auto arange<T>(T, T, T) -> nd_array<T>

TF_CPP_INSTANTIATE_RANGE(std::int32_t);
TF_CPP_INSTANTIATE_RANGE(std::int64_t);
TF_CPP_INSTANTIATE_RANGE(float);
TF_CPP_INSTANTIATE_RANGE(double);

#undef TF_CPP_INSTANTIATE_RANGE

template auto linspace<float>(float, float, int) -> nd_array<float>;
template auto linspace<double>(double, double, int) -> nd_array<double>;

#define TF_CPP_INSTANTIATE_RANDOM(T)                                           \
  template auto random<T>(tf::small_vector<int, 3>, T, T) -> nd_array<T>

TF_CPP_INSTANTIATE_RANDOM(std::int32_t);
TF_CPP_INSTANTIATE_RANDOM(std::int64_t);
TF_CPP_INSTANTIATE_RANDOM(float);
TF_CPP_INSTANTIATE_RANDOM(double);

#undef TF_CPP_INSTANTIATE_RANDOM

} // namespace tf::cpp
