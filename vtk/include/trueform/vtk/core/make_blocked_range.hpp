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
#include <trueform/vtk/core/make_range.hpp>

namespace tf::vtk {

// Static blocked range types
template <std::size_t V>
using float_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<float_range_t>()));

template <std::size_t V>
using double_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<double_range_t>()));

template <std::size_t V>
using int_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<int_range_t>()));

template <std::size_t V>
using vtkIdType_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<vtkIdType_range_t>()));

template <std::size_t V>
using signed_char_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<signed_char_range_t>()));

template <std::size_t V>
using unsigned_char_blocked_range_t =
    decltype(tf::make_blocked_range<V>(std::declval<unsigned_char_range_t>()));

// Dynamic blocked range types
using float_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<float_range_t>(), std::size_t{}));

using double_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<double_range_t>(), std::size_t{}));

using int_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<int_range_t>(), std::size_t{}));

using vtkIdType_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<vtkIdType_range_t>(), std::size_t{}));

using signed_char_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<signed_char_range_t>(), std::size_t{}));

using unsigned_char_dynamic_blocked_range_t =
    decltype(tf::make_blocked_range(std::declval<unsigned_char_range_t>(), std::size_t{}));

/// @brief Creates a blocked range view over vtkFloatArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkFloatArray *array) -> float_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkFloatArray (dynamic block size).
auto make_blocked_range(vtkFloatArray *array, std::size_t block_size)
    -> float_dynamic_blocked_range_t;

/// @brief Creates a blocked range view over vtkDoubleArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkDoubleArray *array) -> double_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkDoubleArray (dynamic block size).
auto make_blocked_range(vtkDoubleArray *array, std::size_t block_size)
    -> double_dynamic_blocked_range_t;

/// @brief Creates a blocked range view over vtkIntArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkIntArray *array) -> int_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkIntArray (dynamic block size).
auto make_blocked_range(vtkIntArray *array, std::size_t block_size)
    -> int_dynamic_blocked_range_t;

/// @brief Creates a blocked range view over vtkIdTypeArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkIdTypeArray *array) -> vtkIdType_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkIdTypeArray (dynamic block size).
auto make_blocked_range(vtkIdTypeArray *array, std::size_t block_size)
    -> vtkIdType_dynamic_blocked_range_t;

/// @brief Creates a blocked range view over vtkSignedCharArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkSignedCharArray *array) -> signed_char_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkSignedCharArray (dynamic block size).
auto make_blocked_range(vtkSignedCharArray *array, std::size_t block_size)
    -> signed_char_dynamic_blocked_range_t;

/// @brief Creates a blocked range view over vtkUnsignedCharArray (static block size).
template <std::size_t V>
auto make_blocked_range(vtkUnsignedCharArray *array) -> unsigned_char_blocked_range_t<V> {
  return tf::make_blocked_range<V>(make_range(array));
}

/// @brief Creates a blocked range view over vtkUnsignedCharArray (dynamic block size).
auto make_blocked_range(vtkUnsignedCharArray *array, std::size_t block_size)
    -> unsigned_char_dynamic_blocked_range_t;

} // namespace tf::vtk
