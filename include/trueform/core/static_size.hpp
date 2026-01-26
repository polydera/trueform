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
#include <array>
#include <cstddef>

namespace tf {
/// @ingroup core_ranges
/// @brief Sentinel value representing dynamic (runtime-determined) size.
///
/// This constant is used throughout the library to indicate that a range or structure
/// does not have a known compile-time size.
///
/// For example, if `tf::static_size<T>` is not specialized, it will default to this value.
///
/// @see @ref tf::static_size
/// @see @ref tf::static_size_v
static constexpr std::size_t dynamic_size = static_cast<std::size_t>(-1);

/// @ingroup core_ranges
/// @brief Compile-time trait that yields the static size of a type, if known.
///
/// By default, this trait evaluates to @ref tf::dynamic_size unless specialized
/// for a specific type. You can specialize `tf::static_size<T>` to expose a
/// compile-time size for your custom container types or views.
///
/// Specializations must inherit from `std::integral_constant<std::size_t, N>`,
/// where `N` is the fixed size.
///
/// @tparam T The type whose static size is being queried.
/// @see @ref tf::static_size_v
template <typename T>
struct static_size : std::integral_constant<std::size_t, tf::dynamic_size> {};

/// @ingroup core_ranges
/// @brief Convenience alias for accessing the value of @ref tf::static_size.
///
/// This is equivalent to `tf::static_size<T>::value`, and is provided for cleaner
/// syntax and compatibility with modern C++ idioms.
///
/// @tparam T The type whose static size is being queried.
/// @see @ref tf::static_size
template <typename T>
static constexpr std::size_t static_size_v = static_size<std::decay_t<T>>::value;

/// @ingroup core_ranges
/// @brief Specialization of @ref tf::static_size for `std::array`.
///
/// This trait exposes the compile-time size `N` for `std::array<T, N>`
/// so that size-aware algorithms and views can propagate it automatically.
///
/// @tparam T The element type.
/// @tparam N The static number of elements in the array.
template <typename T, std::size_t N>
struct static_size<std::array<T, N>> : std::integral_constant<std::size_t, N> {
};
} // namespace tf
