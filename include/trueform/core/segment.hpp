/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./base/seg.hpp"
#include "./coordinate_type.hpp"
#include "./point.hpp"
#include "./static_size.hpp"
namespace tf {
/**
 * @ingroup geometry
 * @brief The line_segment class wrapper
 *
 * ### Policy
 *
 * Policy defines the implementation of the
 * line_segment. It must define:
 *
 * * `Policy::operator[]`: returns a point
 * * `Policy::begin()`: returns an iterator
 *   to the begining of the point range
 * * `Policy::end()`: returns an iterator
 *   to the ending of the point range
 *
 * @tparam Policy The policy that defines the
 * implementation of the line_segment
 */

template <std::size_t Dims, typename Policy> class segment : public Policy {
private:
  using base_t = Policy;

public:
  segment(const Policy &policy) : base_t{policy} {}
  segment(Policy &&policy) : base_t{std::move(policy)} {}
  segment() = default;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;

  friend auto unwrap(const segment &seg) -> decltype(auto) {
    return static_cast<const Policy &>(seg);
  }

  friend auto unwrap(segment &seg) -> decltype(auto) {
    return static_cast<Policy &>(seg);
  }

  friend auto unwrap(segment &&seg) -> decltype(auto) {
    return static_cast<Policy &&>(seg);
  }

  template <typename T> friend auto wrap_like(const segment &, T &&t) {
    return segment<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(segment &, T &&t) {
    return segment<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(segment &&, T &&t) {
    return segment<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }
};

template <std::size_t I, std::size_t Dims, typename Policy>
auto get(const tf::segment<Dims, Policy> &t) -> decltype(auto) {
  return t[I];
}

template <std::size_t I, std::size_t Dims, typename Policy>
auto get(tf::segment<Dims, Policy> &t) -> decltype(auto) {
  return t[I];
}

template <std::size_t I, std::size_t Dims, typename Policy>
auto get(tf::segment<Dims, Policy> &&t) -> decltype(auto) {
  return t[I];
}

} // namespace tf

namespace std {

template <std::size_t I, std::size_t Dims, typename Policy>
struct tuple_element<I, tf::segment<Dims, Policy>> : tuple_element<I, Policy> {
};

template <std::size_t Dims, typename Policy>
struct tuple_size<tf::segment<Dims, Policy>>
    : std::integral_constant<std::size_t, 2> {};

} // namespace std

namespace tf {

template <std::size_t Dims, typename Policy>
struct static_size<tf::segment<Dims, Policy>>
    : std::integral_constant<std::size_t, 2> {};

/// @ingroup geometry
/// @brief Constructs a segment by indirectly indexing into a point range.
///
/// This overload creates a @ref tf::segment by using a range of indices (`ids`)
/// to select elements from a range of points (`points`). The result is a
/// segment consisting of the points at the specified indices, in order.
///
/// @tparam Range0 A range type representing the indices (e.g., a container of
/// integers).
/// @tparam Range1 A range type representing the source points (e.g.,
/// `std::vector<vec2>`).
/// @param ids A range of indices referencing elements in the `points` range.
/// @param points A range of point data from which the segment will be
/// constructed.
/// @return A @ref tf::segment containing the selected points.
///
/// @note Internally uses @ref tf::make_indirect_range to perform indirection.
/// Hence `.ids()` will be accessible.
template <typename Range0, typename Range1>
auto make_segment(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_seg(static_cast<Range0 &&>(ids),
                                   static_cast<Range1 &&>(points));
  return tf::segment<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup geometry
/// @brief Constructs a segment directly from a point range.
///
/// This overload creates a @ref tf::segment by directly forwarding a range of
/// points.
///
/// @tparam Range A range type containing point elements (e.g.,
/// `std::array<vec2, 2>` or `std::vector<vec2>`).
/// @param points A range of points to be included in the segment.
/// @return A @ref tf::segment constructed directly from the input range.
template <typename Range> auto make_segment(Range &&points) {
  auto policy = tf::core::make_seg(static_cast<Range &&>(points));
  return tf::segment<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup geometry
/// @brief Constructs a segment between two points.
template <std::size_t Dims, typename T0, typename T1>
auto make_segment_between_points(const tf::point_like<Dims, T0> &pt0,
                                 const tf::point_like<Dims, T1> &pt1) {
  using pt_t = tf::point<tf::coordinate_type<T0, T1>, Dims>;
  return make_segment(std::array<pt_t, 2>{pt_t{pt0}, pt_t{pt1}});
}

} // namespace tf
