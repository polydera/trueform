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
#include "./tuple.hpp"
#include <type_traits>

namespace tf::core {

namespace detail {
template <typename, typename, typename = std::void_t<>>
struct is_transformable : std::false_type {};

template <typename T, typename U>
struct is_transformable<
    T, U,
    std::void_t<decltype(transformed(std::declval<T>(), std::declval<U>()))>>
    : std::true_type {};

template <typename... Ts, typename U>
struct is_transformable<tf::tuple<Ts...>, U>
    : std::integral_constant<bool,
                             (... || detail::is_transformable<Ts, U>::value)> {
};
} // namespace detail

template <typename T, typename U>
inline constexpr bool is_transformable =
    detail::is_transformable<std::decay_t<T>, U>::value;
} // namespace tf::core
