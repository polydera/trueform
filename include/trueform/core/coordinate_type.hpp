/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <type_traits>

namespace tf {

namespace core {

// Check if T has a nested type named coordinate_type
template <typename, typename = void>
struct has_coordinate_type : std::false_type {};

template <typename T>
struct has_coordinate_type<T, std::void_t<typename T::coordinate_type>>
    : std::true_type {};

// Check if T has a nested type named value_type
template <typename, typename = void> struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {
};

// Primary template
template <typename T, typename = void> struct coordinate_type_deducer {
  using type = std::conditional_t<std::is_fundamental_v<T>, T,
                                  void // fallback if no match
                                  >;
};

// Specialization for types with coordinate_type
template <typename T>
struct coordinate_type_deducer<
    T, std::enable_if_t<core::has_coordinate_type<T>::value>> {
  using type = typename T::coordinate_type;
};

// Specialization for types with value_type (and no coordinate_type)
template <typename T>
struct coordinate_type_deducer<
    T, std::enable_if_t<!core::has_coordinate_type<T>::value &&
                        core::has_value_type<T>::value>> {
  using type = typename coordinate_type_deducer<
      std::decay_t<typename T::value_type>>::type;
};
} // namespace core

// Alias
template <typename T, typename... Ts>
using coordinate_type = std::common_type_t<
    typename core::coordinate_type_deducer<std::decay_t<T>>::type,
    typename core::coordinate_type_deducer<std::decay_t<Ts>>::type...>;

} // namespace tf
