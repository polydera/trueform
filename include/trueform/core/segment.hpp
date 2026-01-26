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

#include "./base/seg.hpp"
#include "./coordinate_type.hpp"
#include "./point.hpp"
#include "./static_size.hpp"
namespace tf {
/// @ingroup core_primitives
/// @brief A line segment connecting two points.
///
/// A segment represents a finite portion of a line between two endpoints.
/// It provides iteration and element access to its two points via `operator[]`.
///
/// Use `tf::make_segment()` or `tf::make_segment_between_points()` for
/// construction.
///
/// @tparam Dims The dimensionality (e.g., 2, 3).
/// @tparam Policy The policy that defines the storage implementation.

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

/// @ingroup core_primitives
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

/// @ingroup core_primitives
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

/// @ingroup core_primitives
/// @brief Constructs a segment between two points.
///
/// Creates a segment connecting the two given points.
///
/// @tparam Dims The dimensionality.
/// @tparam T0 Policy type for the first point.
/// @tparam T1 Policy type for the second point.
/// @param pt0 The first endpoint.
/// @param pt1 The second endpoint.
/// @return A segment connecting pt0 and pt1.
template <std::size_t Dims, typename T0, typename T1>
auto make_segment_between_points(const tf::point_like<Dims, T0> &pt0,
                                 const tf::point_like<Dims, T1> &pt1) {
  using pt_t = tf::point<tf::coordinate_type<T0, T1>, Dims>;
  return make_segment(std::array<pt_t, 2>{pt_t{pt0}, pt_t{pt1}});
}

} // namespace tf
