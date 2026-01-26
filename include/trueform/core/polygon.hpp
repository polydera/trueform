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

#include "./base/poly.hpp"
#include "./static_size.hpp"
namespace tf {
/// @ingroup core_primitives
/// @brief A polygon defined by a sequence of vertices.
///
/// A polygon represents a closed shape defined by its vertices. The number
/// of vertices can be fixed at compile time or dynamic (`tf::dynamic_size`).
///
/// Use `tf::make_polygon()` for construction from points or indices+points.
///
/// @tparam Dims The dimensionality of the space (e.g., 2, 3).
/// @tparam Policy The policy that defines the storage implementation.
template <std::size_t Dims, typename Policy> class polygon : public Policy {
private:
  using base_t = Policy;

public:
  polygon(const Policy &policy) : base_t{policy} {}
  polygon(Policy &&policy) : base_t{std::move(policy)} {}
  polygon() = default;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;

  friend auto unwrap(const polygon &seg) -> decltype(auto) {
    return static_cast<const Policy &>(seg);
  }

  friend auto unwrap(polygon &seg) -> decltype(auto) {
    return static_cast<Policy &>(seg);
  }

  friend auto unwrap(polygon &&seg) -> decltype(auto) {
    return static_cast<Policy &&>(seg);
  }

  template <typename T> friend auto wrap_like(const polygon &, T &&t) {
    return polygon<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(polygon &, T &&t) {
    return polygon<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(polygon &&, T &&t) {
    return polygon<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }
};

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(const tf::polygon<Dims, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(tf::polygon<Dims, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(tf::polygon<Dims, Policy> &&t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t Dims, typename Policy>
struct static_size<tf::polygon<Dims, Policy>> : static_size<Policy> {};

/// @ingroup core_primitives
/// @brief Constructs a fixed-size polygon by indirectly indexing into points.
///
/// @tparam V The number of vertices (compile-time constant).
/// @tparam Range0 A range type for indices.
/// @tparam Range1 A range type for points.
/// @param ids Indices referencing elements in the points range.
/// @param points The source points.
/// @return A polygon with V vertices.
template <std::size_t V, typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_poly<V>(static_cast<Range0 &&>(ids),
                                       static_cast<Range1 &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Constructs a fixed-size polygon from a point range.
///
/// @tparam V The number of vertices (compile-time constant).
/// @tparam Range A range type for points.
/// @param points The points defining the polygon.
/// @return A polygon with V vertices.
template <std::size_t V, typename Range> auto make_polygon(Range &&points) {
  auto policy = tf::core::make_poly<V>(static_cast<Range &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Constructs a polygon by indirectly indexing into points.
///
/// The number of vertices is deduced from @ref tf::static_size of the ids range.
///
/// @tparam Range0 A range type for indices.
/// @tparam Range1 A range type for points.
/// @param ids Indices referencing elements in the points range.
/// @param points The source points.
/// @return A polygon.
template <typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_poly(static_cast<Range0 &&>(ids),
                                    static_cast<Range1 &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Constructs a polygon from a point range.
///
/// The number of vertices is deduced from @ref tf::static_size of the range.
///
/// @tparam Range A range type for points.
/// @param points The points defining the polygon.
/// @return A polygon.
template <typename Range> auto make_polygon(Range &&points) {
  auto policy = tf::core::make_poly(static_cast<Range &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

} // namespace tf
namespace std {
template <std::size_t Dims, typename Policy>
struct tuple_size<tf::polygon<Dims, Policy>> : tuple_size<Policy> {};

template <std::size_t I, std::size_t Dims, typename Policy>
struct tuple_element<I, tf::polygon<Dims, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<tf::polygon<Dims, Policy>>().begin())>::value_type;
};

template <std::size_t I, typename Policy>
struct tuple_element<I, tf::polygon<tf::dynamic_size, Policy>>;

} // namespace std
