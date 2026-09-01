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

template <typename, typename, typename = std::void_t<>>
struct has_transformed : std::false_type {};

template <typename T, typename U>
struct has_transformed<
    T, U,
    std::void_t<decltype(transformed(std::declval<T>(), std::declval<U>()))>>
    : std::true_type {};

template <typename... Ts, typename U>
struct has_transformed<tf::tuple<Ts...>, U>
    : std::integral_constant<bool, (... || has_transformed<Ts, U>::value)> {};

template <typename T, typename U>
inline constexpr bool is_transformable =
    has_transformed<std::decay_t<T>, U>::value;
} // namespace tf::core
