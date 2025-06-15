/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./indirect_range.hpp"
#include "./inject_ids.hpp"
#include "./point.hpp"
#include "./static_size.hpp"
#include "./value_type.hpp"
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

template <typename Policy> class segment : public Policy {
private:
  using base_t = Policy;

public:
  segment(const Policy &policy) : base_t{policy} {}
  segment(Policy &&policy) : base_t{std::move(policy)} {}
  segment() = default;
  using base_t::base_t;
  using base_t::operator=;

  /**
   * @brief Indexes into the points of the segment
   *
   * @param i Index
   * @return Point at index `i`
   */
  auto operator[](std::size_t i) -> decltype(auto) {
    return base_t::operator[](i);
  }
  /**
   * @brief Indexes into the points of the segment
   *
   * @param i Index
   * @return Point at index `i`
   */
  auto operator[](std::size_t n) const -> decltype(auto) {
    return base_t::operator[](n);
  }

  /**
   * @brief Returns the iterator to the begining of
   * the point range
   *
   * @return An iterator
   */
  auto begin() const { return base_t::begin(); }
  /**
   * @brief Returns the iterator to the begining of
   * the point range
   *
   * @return An iterator
   */
  auto begin() { return base_t::begin(); }
  /**
   * @brief Returns the iterator to the ending of
   * the point range
   *
   * @return An iterator
   */
  auto end() const { return base_t::end(); }
  /**
   * @brief Returns the iterator to the ending of
   * the point range
   *
   * @return An iterator
   */
  auto end() { return base_t::end(); }
  /**
   * @brief Returns the number of points
   *
   * @return Number of points
   */
  constexpr auto size() const { return 2; }
};

template <std::size_t I, typename Policy>
auto get(const tf::segment<Policy> &t) -> decltype(auto) {
  return t[I];
}

template <std::size_t I, typename Policy>
auto get(tf::segment<Policy> &t) -> decltype(auto) {
  return t[I];
}

template <typename Policy>
struct static_size<tf::segment<Policy>>
    : std::integral_constant<std::size_t, 2> {};

template <typename Policy>
auto unwrap(const segment<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(segment<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(segment<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const segment<Policy> &, T &&t) {
  return segment<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segment<Policy> &, T &&t) {
  return segment<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segment<Policy> &&, T &&t) {
  return segment<std::decay_t<T>>{static_cast<T &&>(t)};
}

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
  auto policy = tf::make_indirect_range(static_cast<Range0 &&>(ids),
                                        static_cast<Range1 &&>(points));
  return tf::segment<decltype(policy)>(tf::make_indirect_range(
      static_cast<Range0 &&>(ids), static_cast<Range1 &&>(points)));
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
  return tf::segment<std::decay_t<Range>>(static_cast<Range &&>(points));
}

/// @ingroup geometry
/// @brief Constructs a segment between two points.
template <std::size_t Dims, typename T0, typename T1>
auto make_segment_between_points(const tf::point_like<Dims, T0> &pt0,
                                 const tf::point_like<Dims, T1> &pt1) {
  using pt_t = tf::point<tf::common_value<T0, T1>, Dims>;
  return tf::segment<std::array<pt_t, 2>>(
      std::array<pt_t, 2>{pt_t{pt0}, pt_t{pt1}});
}

} // namespace tf
