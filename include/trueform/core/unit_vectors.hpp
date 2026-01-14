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

#include "./views/mapped_range.hpp"
#include "./views/unit_vectors.hpp"

namespace tf {

namespace core {
template <typename T> struct unit_vec_as_dref {
  template <std::size_t Dims, typename Policy>
  auto operator()(unit_vector_like<Dims, Policy> &pt) const {
    return pt.template as<T>();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(const unit_vector_like<Dims, Policy> &pt) const {
    return pt.template as<T>();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(unit_vector_like<Dims, Policy> &&pt) const {
    return pt.template as<T>();
  }
};
} // namespace core

template <typename Policy> struct unit_vectors : Policy {
  unit_vectors(const Policy &r) : Policy{r} {}
  unit_vectors(Policy &&r) : Policy{std::move(r)} {}

  template <typename T> auto as() const {
    auto r = tf::make_mapped_range(*this, core::unit_vec_as_dref<T>{});
    return unit_vectors<decltype(r)>{r};
  }

  template <typename T> auto as() {
    auto r = tf::make_mapped_range(*this, core::unit_vec_as_dref<T>{});
    return unit_vectors<decltype(r)>{r};
  }
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

/// @ingroup core_ranges
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
template <std::size_t Dims, typename Range> auto make_unit_vectors(Range &&r) {
  auto pts = tf::views::make_unit_vectors<Dims>(r);
  return tf::unit_vectors<decltype(pts)>{pts};
}

template <typename Range> auto make_unit_vectors(Range &&r) {
  auto vec = tf::make_range(r);
  return tf::unit_vectors<decltype(vec)>{vec};
}

template <typename Range>
auto make_unit_vectors(unit_vectors<Range> r) -> unit_vectors<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::unit_vectors<Policy> &obj) {
  return obj;
}
} // namespace tf
