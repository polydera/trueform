/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./implementation/unit_vector_iterator.hpp"
#include "./range.hpp"

namespace tf {

namespace implementation {
template <std::size_t Dims, typename Range>
auto make_unit_vectors(Range &&r) {
  auto begin =
      tf::implementation::iter::make_unit_vector_iterator<Dims>(r.begin());
  auto end = tf::implementation::iter::make_unit_vector_iterator<Dims>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}
} // namespace implementation

template <typename Policy> struct unit_vectors : Policy {
  unit_vectors(const Policy &r) : Policy{r} {}
  unit_vectors(Policy &&r) : Policy{r} {}
};

template <typename Policy>
auto unwrap(const unit_vectors<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(unit_vectors<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(unit_vectors<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const unit_vectors<Policy> &, T &&t) {
  return unit_vectors<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(unit_vectors<Policy> &, T &&t) {
  return unit_vectors<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(unit_vectors<Policy> &&, T &&t) {
  return unit_vectors<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup ranges
/// @brief Creates a range of unit_vectors from a flat scalar sequence.
///
/// This utility interprets a flat range of scalars as a sequence of
/// fixed-dimensional unit_vectors. It constructs a @ref tf::range view over
/// `Dims`-dimensional @ref tf::vector_view elements, where each unit_vector
/// occupies `Dims` consecutive scalars in the original range.
///
/// This is especially useful when working with flat buffers of interleaved
/// coordinates, such as geometry loaded from binary files or raw memory
/// layouts.
///
/// @tparam Dims The number of dimensions per unit_vector (e.g., 2 or 3).
/// @tparam Range A range type whose elements are scalar values (e.g., float,
/// double).
/// @param r A flat range of scalar values representing interleaved unit_vector
/// coordinates.
/// @return A @ref tf::range of @ref tf::vector_view elements, each representing
/// a unit_vector.
///
/// @note The size of the returned range is `r.size() / Dims`.
/// @note The input range must contain a total number of elements divisible by
/// `Dims`.
///
/// @code
/// tf::buffer<float> flat{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
/// for (auto pt : make_unit_vectors_range<3>(flat)) {
///   auto [x, y, z] = pt;
///   std::cout << x << ", " << y << ", " << z << '\n';
/// }
/// // Output:
/// // 1, 2, 3,
/// // 4, 5, 6
/// @endcode
template <std::size_t Dims, typename Range>
auto make_unit_vectors(Range &&r) {
  auto pts = tf::implementation::make_unit_vectors<3>(r);
  return tf::unit_vectors<decltype(pts)>{pts};
}

template <typename Range> auto make_unit_vectors(Range &&r) {
  return tf::unit_vectors<std::decay_t<Range>>{r};
}

template <typename Range>
auto make_unit_vectors(unit_vectors<Range> r)
    -> unit_vectors<Range> {
  return r;
}
} // namespace tf
