/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./indirect_range.hpp"
#include "./inject_id.hpp"
#include "./inject_ids.hpp"
#include "./inject_normal.hpp"
#include "./inject_plane.hpp"
#include "./static_size.hpp"
namespace tf {
/**
 * @ingroup geometry
 * @brief Base class for the polygon
 *
 * ### Policy
 *
 * Policy defines the implementation of the
 * polygon. It must define:
 *
 * * `Policy::operator[]`: returns a point
 * * `Policy::begin()`: returns an iterator
 *   to the begining of the point range
 * * `Policy::end()`: returns an iterator
 *   to the ending of the point range
 * * `Policy::size()`: returns the number
 *   of points
 *
 * @tparam V Number of vertices, can be tf::dynamic_size
 * @tparam Policy The policy that defines the
 * implementation of the polygon
 */
template <std::size_t V, typename Policy> class polygon : public Policy {
private:
  using base_t = Policy;

public:
  polygon(const Policy &policy) : base_t{policy} {}
  polygon(Policy &&policy) : base_t{std::move(policy)} {}
  polygon() = default;
  using base_t::base_t;
  using base_t::operator=;

  /**
   * @brief Indexes into the points of the polygon
   *
   * @param i Index
   * @return Point at index `i`
   */
  auto operator[](std::size_t i) -> decltype(auto) {
    return base_t::operator[](i);
  }
  /**
   * @brief Indexes into the points of the polygon
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
  auto size() const { return base_t::size(); }
};

template <std::size_t I, std::size_t V, typename Policy,
          typename = std::enable_if_t<V != tf::dynamic_size, void>>
auto get(const tf::polygon<V, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t I, std::size_t V, typename Policy,
          typename = std::enable_if_t<V != tf::dynamic_size, void>>
auto get(tf::polygon<V, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<Policy &>(t));
}

template <std::size_t I, std::size_t V, typename Policy,
          typename = std::enable_if_t<V != tf::dynamic_size, void>>
auto get(tf::polygon<V, Policy> &&t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<Policy &&>(t));
}

template <std::size_t V, typename Policy>
struct static_size<tf::polygon<V, Policy>>
    : std::integral_constant<std::size_t, V> {};

template <std::size_t V, typename Policy>
auto unwrap(const polygon<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(polygon<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(polygon<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const polygon<V, Policy> &, T &&t) {
  return polygon<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(polygon<V, Policy> &, T &&t) {
  return polygon<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(polygon<V, Policy> &&, T &&t) {
  return polygon<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const polygon<V, Policy> &&, T &&t) {
  return polygon<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup geometry
/// @brief Constructs a polygon by indirectly indexing into a point range.
///
/// This function creates a polygon by applying an index indirection:
/// it takes a range of indices (`ids`) and a corresponding range of points
/// (`points`), and returns a @ref tf::polygon that consists of the points at
/// the specified indices.
///
/// The function preserves and propagates static size information from the `ids`
/// range, enabling potential compile-time optimizations for fixed-size
/// polygons.
///
/// @tparam Range0 A range type representing the indices (typically a container
/// of integers).
/// @tparam Range1 A range type representing the source points (e.g.,
/// `std::vector<vec2>`).
/// @param ids A range of indices referencing the positions in the `points`
/// range.
/// @param points A range of point data from which the polygon will be
/// constructed.
/// @return A @ref tf::polygon containing the selected points in the order
/// specified by `ids`.
///
/// @note If `Range0` has a known static size, it will be encoded in the
/// resulting polygon type.
/// @note Internally uses @ref tf::make_indirect_range to perform indirection.
/// Hence `.ids()` will be accessible.

template <typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::make_indirect_range(static_cast<Range0 &&>(ids),
                                        static_cast<Range1 &&>(points));
  return tf::polygon<tf::static_size_v<Range0>, decltype(policy)>(
      std::move(policy));
}

template <std::size_t V, typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::make_indirect_range(static_cast<Range0 &&>(ids),
                                        static_cast<Range1 &&>(points));
  return tf::polygon<V, decltype(policy)>(std::move(policy));
}

/// @ingroup geometry
/// @brief Constructs a polygon directly from a point range.
///
/// This function creates a @ref tf::polygon by directly forwarding a range of
/// points.
///
/// Static size information from the input range is propagated to the resulting
/// polygon type, enabling compile-time optimizations when the size is known.
///
/// @tparam Range A range type containing point elements (e.g.,
/// `std::array<vec2, N>` or `std::vector<vec2>`).
/// @param points A range of points to be included in the polygon.
/// @return A @ref tf::polygon constructed directly from the input range.
///
/// @note The polygon will use @ref tf::static_size_v to encode static size
/// information, if available.

template <typename Range> auto make_polygon(Range &&points) {
  return tf::polygon<tf::static_size_v<Range>, std::decay_t<Range>>(
      static_cast<Range &&>(points));
}

template <std::size_t V, typename Range> auto make_polygon(Range &&points) {
  return tf::polygon<V, std::decay_t<Range>>(static_cast<Range &&>(points));
}

template <std::size_t V, typename Policy>
auto inject_plane(const polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_injected_plane<Policy>) {
    return poly;
  } else if constexpr (has_injected_normal<Policy>) {
    return tf::inject_plane(tf::make_plane(poly.normal(), poly[0]), poly);
  } else {
    return tf::inject_plane(tf::make_plane(poly[0], poly[1], poly[2]), poly);
  }
}

template <std::size_t V, typename Policy>
auto inject_plane(polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_injected_plane<Policy>) {
    return poly;
  } else if constexpr (has_injected_normal<Policy>) {
    return tf::inject_plane(tf::make_plane(poly.normal(), poly[0]), poly);
  } else {
    return tf::inject_plane(tf::make_plane(poly[0], poly[1], poly[2]),
                            poly);
  }
}

template <std::size_t V, typename Policy>
auto inject_normal(const polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_injected_plane<Policy> || has_injected_normal<Policy>) {
    return poly;
  } else {
    return tf::inject_normal(tf::normal(poly[0], poly[1], poly[2]), poly);
  }
}

template <std::size_t V, typename Policy>
auto inject_normal(polygon<V, Policy> &poly) -> decltype(auto) {
  if constexpr (has_injected_plane<Policy> || has_injected_normal<Policy>) {
    return poly;
  } else {
    return tf::inject_normal(tf::normal(poly[0], poly[1], poly[2]), poly);
  }
}

template <typename Range0, typename Range1, std::size_t Dims, typename T>
auto make_polygon(Range0 &&ids, Range1 &&points,
                  const tf::unit_vector_like<Dims, T> &normal) {
  return tf::inject_normal(normal,
                           tf::make_polygon(static_cast<Range0 &&>(ids),
                                            static_cast<Range1 &&>(points)));
}

} // namespace tf
namespace std {
template <std::size_t V, typename Policy>
struct tuple_size<tf::polygon<V, Policy>>
    : std::integral_constant<std::size_t, V> {};

template <typename Policy>
struct tuple_size<tf::polygon<tf::dynamic_size, Policy>>;

template <std::size_t I, std::size_t V, typename Policy>
struct tuple_element<I, tf::polygon<V, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<tf::polygon<V, Policy>>().begin())>::value_type;
};

template <std::size_t I, typename Policy>
struct tuple_element<I, tf::polygon<tf::dynamic_size, Policy>>;

} // namespace std
