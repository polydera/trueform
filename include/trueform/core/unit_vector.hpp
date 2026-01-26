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
#include "./coordinate_type.hpp"
#include "./unit_vector_like.hpp"
#include "./unsafe.hpp"
#include "./vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A fixed-size unit vector wrapper type.
///
/// `unit_vector<T, N>` represents a vector of dimension `N` with a fixed length
/// of 1. It inherits from `tf::vector<T, N>` but provides guarantees and
/// overrides to reflect the unit-length invariant. All instances must be
/// normalized at construction time.
///
/// Use `make_unit_vector()` or `make_unit_vector_unsafe()` to create instances.
///
/// @tparam T The scalar type (e.g. float, double).
/// @tparam N The number of dimensions (e.g. 2, 3, 4).
template <typename T, std::size_t Dims>
using unit_vector = tf::unit_vector_like<Dims, core::vec<T, Dims>>;

/// @ingroup core_primitives
/// @brief Safely construct a unit vector by normalizing the input.
///
/// This function creates a `unit_vector<T, Dims>` from any vector-like input
/// by computing its normalized form.
///
/// @tparam Dims The number of dimensions.
/// @tparam T A type derived from `vector_like`.
/// @param v A vector that will be normalized before being wrapped.
/// @return A `unit_vector` instance with length 1.
template <std::size_t Dims, typename T>
auto make_unit_vector(const tf::vector_like<Dims, T> &v) {
  return unit_vector<tf::coordinate_type<T>, Dims>{v};
}

/// @overload
template <std::size_t Dims, typename T>
auto make_unit_vector(const tf::unit_vector_like<Dims, T> &v) {
  return unit_vector<tf::coordinate_type<T>, Dims>{v};
}

/// @ingroup core_primitives
/// @brief Construct a unit vector from an already-normalized input.
///
/// This function creates a `unit_vector<T, Dims>` assuming the input is already
/// normalized. No normalization is performed. Use only if you are sure of the
/// input.
///
/// @tparam Dims The number of dimensions.
/// @tparam T A type derived from `vector_like`.
/// @param v A vector with unit length.
/// @return A `unit_vector` instance wrapping the given vector.
template <std::size_t Dims, typename T>
auto make_unit_vector(tf::unsafe_t, const tf::vector_like<Dims, T> &v) {
  return unit_vector<tf::coordinate_type<T>, Dims>{tf::unsafe, v};
}

/// @ingroup core_primitives
/// @brief Construct a unit vector from individual coordinate values (normalized).
///
/// Creates a @ref tf::unit_vector by deducing type and dimensionality from
/// the provided arguments. The resulting vector is normalized to unit length.
/// Requires at least 2 coordinates.
///
/// @tparam T The coordinate type (deduced from first two arguments).
/// @tparam Ts Additional coordinate types.
/// @param t0 The first coordinate value.
/// @param t1 The second coordinate value.
/// @param ts Additional coordinate values.
/// @return A normalized `tf::unit_vector<common_type, N>` where N = 2 + sizeof...(ts).
template <typename T, typename... Ts>
auto make_unit_vector(const T &t0, const T &t1, const Ts &...ts)
    -> tf::unit_vector<std::common_type_t<T, Ts...>, (2 + sizeof...(Ts))> {
  using type = std::common_type_t<T, Ts...>;
  constexpr std::size_t Dims = 2 + sizeof...(Ts);
  return unit_vector<type, Dims>{
      tf::vector<type, Dims>{static_cast<type>(t0), static_cast<type>(t1),
                             static_cast<type>(ts)...}};
}

/// @ingroup core_primitives
/// @brief Construct a unit vector from individual coordinate values (unsafe).
///
/// Creates a @ref tf::unit_vector by deducing type and dimensionality from
/// the provided arguments. No normalization is performed - the input must
/// already have unit length. Requires at least 2 coordinates.
///
/// @tparam T The coordinate type (deduced from first two arguments).
/// @tparam Ts Additional coordinate types.
/// @param t0 The first coordinate value.
/// @param t1 The second coordinate value.
/// @param ts Additional coordinate values.
/// @return A `tf::unit_vector<common_type, N>` where N = 2 + sizeof...(ts).
template <typename T, typename... Ts>
auto make_unit_vector(tf::unsafe_t, const T &t0, const T &t1, const Ts &...ts)
    -> tf::unit_vector<std::common_type_t<T, Ts...>, (2 + sizeof...(Ts))> {
  using type = std::common_type_t<T, Ts...>;
  constexpr std::size_t Dims = 2 + sizeof...(Ts);
  return unit_vector<type, Dims>{
      tf::unsafe,
      tf::vector<type, Dims>{static_cast<type>(t0), static_cast<type>(t1),
                             static_cast<type>(ts)...}};
}

} // namespace tf
