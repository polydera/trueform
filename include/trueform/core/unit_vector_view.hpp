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
#include "./vector_view.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A fixed-size unit vector wrapper type.
///
/// `unit_vector_view<T, N>` represents a vector of dimension `N` with a fixed
/// length of 1. It inherits from `tf::vector<T, N>` but provides guarantees and
/// overrides to reflect the unit-length invariant. All instances must be
/// normalized at construction time.
///
/// Use `make_unit_vector_view()` or `make_unit_vector_view_unsafe()` to create
/// instances.
///
/// @tparam T The scalar type (e.g. float, double).
/// @tparam N The number of dimensions (e.g. 2, 3, 4).
template <typename T, std::size_t Dims>
using unit_vector_view = tf::unit_vector_like<Dims, core::vec_view<T, Dims>>;

/// @ingroup core_primitives
/// @brief Safely construct a unit vector by normalizing the input.
///
/// This function creates a `unit_vector_view<T, Dims>` from any vector-like
/// input by computing its normalized form.
///
/// @tparam Dims The number of dimensions.
/// @tparam T A type derived from `vector_like`.
/// @param v A vector that will be normalized before being wrapped.
/// @return A `unit_vector_view` instance with length 1.
template <std::size_t Dims, typename T>
auto make_unit_vector_view(tf::vector_like<Dims, T> v) {
  return unit_vector_view<typename tf::vector_like<Dims, T>::element_type,
                          Dims>{tf::make_vector_view<Dims>(v.data())};
}

template <std::size_t Dims, typename T>
auto make_unit_vector_view(tf::unsafe_t, tf::vector_like<Dims, T> v) {
  return unit_vector_view<typename tf::vector_like<Dims, T>::element_type,
                          Dims>{tf::unsafe,
                                tf::make_vector_view<Dims>(v.data())};
}

template <std::size_t Dims, typename T>
auto make_unit_vector_view(tf::unit_vector_like<Dims, T> v) {
  return make_unit_vector_view(tf::unsafe,
                               tf::make_vector_view<Dims>(v.data()));
}

} // namespace tf
