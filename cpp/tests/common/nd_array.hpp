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

#include "trueform/core/buffer.hpp"
#include "trueform/core/curves_buffer.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp::test {

template <typename T>
auto make_nd_array(std::initializer_list<T> values,
                   tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto &value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

template <typename T>
auto make_nd_array(std::initializer_list<T> values,
                   std::initializer_list<int> shape) -> tf::cpp::nd_array<T> {
  return make_nd_array<T>(values, tf::small_vector<int, 3>(shape));
}

template <typename T>
auto make_nd_array(const std::vector<T> &values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto &value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

template <typename T>
auto make_nd_array(const std::vector<T> &values,
                   std::initializer_list<int> shape) -> tf::cpp::nd_array<T> {
  return make_nd_array(values, tf::small_vector<int, 3>(shape));
}

/// @brief A copy of core's own storage, read as an array.
///
/// Results are core buffers, so a check that reads one by index takes its own
/// copy rather than a view of what it was handed.
template <typename T>
auto copied_nd_array(const tf::buffer<T> &source,
                     tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(source.size());
  std::copy(source.begin(), source.end(), storage.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

template <typename T>
auto make_empty_nd_array(tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(0);
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

template <typename T>
auto make_empty_nd_array(std::initializer_list<int> shape)
    -> tf::cpp::nd_array<T> {
  return make_empty_nd_array<T>(tf::small_vector<int, 3>(shape));
}

template <typename T>
auto has_shape(const tf::cpp::nd_array<T> &array,
               const tf::small_vector<int, 3> &shape) -> bool {
  return array.raw_shape() == shape;
}

template <typename T>
auto has_shape(const tf::cpp::nd_array<T> &array,
               std::initializer_list<int> shape) -> bool {
  return has_shape(array, tf::small_vector<int, 3>(shape));
}

template <typename T>
auto has_values(const tf::cpp::nd_array<T> &array,
                std::initializer_list<T> expected) -> bool {
  if (array.length() != expected.size())
    return false;
  auto actual = array.begin();
  for (const auto &value : expected)
    if (*actual++ != value)
      return false;
  return true;
}

template <typename T>
auto has_values(const tf::cpp::nd_array<T> &array,
                const std::vector<T> &expected) -> bool {
  if (array.length() != expected.size())
    return false;
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (array[index] != expected[index])
      return false;
  return true;
}

template <typename T>
auto all_finite(const tf::cpp::nd_array<T> &array) -> bool {
  if constexpr (!std::is_floating_point<T>::value) {
    return true;
  } else {
    for (const auto value : array)
      if (!std::isfinite(value))
        return false;
    return true;
  }
}

template <typename T>
auto has_independent_storage(const tf::cpp::nd_array<T> &first,
                             const tf::cpp::nd_array<T> &second) -> bool {
  return first.is_valid() && second.is_valid() &&
         first.raw_owner() != second.raw_owner();
}

/// @brief A polyline result, read as the three arrays a check compares.
///
/// A curve result is core's own storage, so a check that reads one by index
/// takes its own copy of the three flat arrays it is — the one place a suite
/// crosses from the buffer to the array carrier.
template <typename Index, typename Real> struct curve_arrays {
  tf::cpp::nd_array<Index> offsets;
  tf::cpp::nd_array<Index> ids;
  tf::cpp::nd_array<Real> points;
};

template <typename Index, typename Real>
auto arrays_of(const tf::curves_buffer<Index, Real, 3> &value)
    -> curve_arrays<Index, Real> {
  const auto &offsets = value.paths_buffer().offsets_buffer();
  const auto &ids = value.paths_buffer().data_buffer();
  const auto &points = value.points_buffer().data_buffer();
  return {copied_nd_array(offsets, {static_cast<int>(offsets.size())}),
          copied_nd_array(ids, {static_cast<int>(ids.size())}),
          copied_nd_array(points, {static_cast<int>(points.size() / 3), 3})};
}

} // namespace tf::cpp::test
