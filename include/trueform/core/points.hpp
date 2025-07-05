/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./range.hpp"
#include "./vectors.hpp"
#include "./views/mapped_range.hpp"
#include "./views/points.hpp"

namespace tf {

namespace core {
struct pt_vec_dref {
  template <std::size_t Dims, typename Policy>
  auto operator()(point_like<Dims, Policy> &pt) const {
    return pt.as_vector_view();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(const point_like<Dims, Policy> &pt) const {
    return pt.as_vector_view();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(point_like<Dims, Policy> &&pt) const {
    return pt.as_vector_view();
  }
};

template <typename T> struct pt_as_dref {
  template <std::size_t Dims, typename Policy>
  auto operator()(point_like<Dims, Policy> &pt) const {
    return pt.template as<T>();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(const point_like<Dims, Policy> &pt) const {
    return pt.template as<T>();
  }

  template <std::size_t Dims, typename Policy>
  auto operator()(point_like<Dims, Policy> &&pt) const {
    return pt.template as<T>();
  }
};
} // namespace core

template <typename Policy> struct points : Policy {
  points(const Policy &r) : Policy{r} {}
  points(Policy &&r) : Policy{std::move(r)} {}

  auto as_vector_view() const {
    auto r = tf::make_mapped_range(*this, core::pt_vec_dref{});
    return vectors<decltype(r)>{r};
  }

  auto as_vector_view() {
    auto r = tf::make_mapped_range(*this, core::pt_vec_dref{});
    return vectors<decltype(r)>{r};
  }

  template <typename T> auto as() const {
    auto r = tf::make_mapped_range(*this, core::pt_as_dref<T>{});
    return points<decltype(r)>{r};
  }

  template <typename T> auto as() {
    auto r = tf::make_mapped_range(*this, core::pt_as_dref<T>{});
    return points<decltype(r)>{r};
  }
};

template <typename Policy>
auto unwrap(const points<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(points<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(points<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const points<Policy> &, T &&t) {
  return points<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(points<Policy> &, T &&t) {
  return points<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(points<Policy> &&, T &&t) {
  return points<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup ranges
/// @brief Creates a range of points from a flat scalar sequence.
///
/// This utility interprets a flat range of scalars as a sequence of
/// fixed-dimensional points. It constructs a @ref tf::range view over
/// `Dims`-dimensional @ref tf::vector_view elements, where each point
/// occupies `Dims` consecutive scalars in the original range.
///
/// This is especially useful when working with flat buffers of interleaved
/// coordinates, such as geometry loaded from binary files or raw memory
/// layouts.
///
/// @tparam Dims The number of dimensions per point (e.g., 2 or 3).
/// @tparam Range A range type whose elements are scalar values (e.g., float,
/// double).
/// @param r A flat range of scalar values representing interleaved point
/// coordinates.
/// @return A @ref tf::range of @ref tf::vector_view elements, each representing
/// a point.
///
/// @note The size of the returned range is `r.size() / Dims`.
/// @note The input range must contain a total number of elements divisible by
/// `Dims`.
///
/// @code
/// tf::buffer<float> flat{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
/// for (auto pt : make_points_range<3>(flat)) {
///   auto [x, y, z] = pt;
///   std::cout << x << ", " << y << ", " << z << '\n';
/// }
/// // Output:
/// // 1, 2, 3,
/// // 4, 5, 6
/// @endcode
template <std::size_t Dims, typename Range> auto make_points(Range &&r) {
  auto pts = tf::views::make_points<Dims>(r);
  return tf::points<decltype(pts)>{pts};
}

template <typename Range> auto make_points(Range &&r) {
  auto pts = tf::make_range(r);
  return tf::points<decltype(pts)>{pts};
}

template <typename Range> auto make_points(points<Range> r) -> points<Range> {
  return r;
}

template <typename Policy> auto make_view(const tf::points<Policy> &obj) {
  return obj;
}
} // namespace tf
