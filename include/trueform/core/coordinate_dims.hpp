/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./coordinate_type.hpp"
#include <type_traits>

namespace tf {

namespace core {

// Check if T has a nested type named coordinate_type
template <typename, typename = void>
struct has_coordinate_dims : std::false_type {};

template <typename T>
struct has_coordinate_dims<T, std::void_t<typename T::coordinate_dims>>
    : std::true_type {};

// Primary template
template <typename T, typename = void>
struct coordinate_dims_deducer : std::integral_constant<std::size_t, 1> {};

// Specialization for types with coordinate_type
template <typename T>
struct coordinate_dims_deducer<
    T, std::enable_if_t<core::has_coordinate_dims<T>::value>> {
  using type = typename T::coordinate_dims;
};

// Specialization for types with value_type (and no coordinate_type)
template <typename T>
struct coordinate_dims_deducer<
    T, std::enable_if_t<!core::has_coordinate_dims<T>::value &&
                        core::has_value_type<T>::value>> {
  using type = typename coordinate_dims_deducer<
      std::decay_t<typename T::value_type>>::type;
};
} // namespace core

// Alias
template <typename T>
using coordinate_dims = typename core::coordinate_dims_deducer<std::decay_t<T>>::type;

template <typename T>
inline constexpr std::size_t coordinate_dims_v = coordinate_dims<T>::value;

} // namespace tf
