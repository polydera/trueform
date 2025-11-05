/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./base/poly.hpp"
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

template <std::size_t V, typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_poly<V>(static_cast<Range0 &&>(ids),
                                       static_cast<Range1 &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

template <std::size_t V, typename Range> auto make_polygon(Range &&points) {
  auto policy = tf::core::make_poly<V>(static_cast<Range &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

template <typename Range0, typename Range1>
auto make_polygon(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_poly(static_cast<Range0 &&>(ids),
                                    static_cast<Range1 &&>(points));
  return tf::polygon<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

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
