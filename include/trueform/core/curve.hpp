/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./base/curve.hpp"
#include "./static_size.hpp"
namespace tf {
template <std::size_t Dims, typename Policy> class curve : public Policy {
private:
  using base_t = Policy;

public:
  curve(const Policy &policy) : base_t{policy} {}
  curve(Policy &&policy) : base_t{std::move(policy)} {}
  curve() = default;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;

  friend auto unwrap(const curve &seg) -> decltype(auto) {
    return static_cast<const Policy &>(seg);
  }

  friend auto unwrap(curve &seg) -> decltype(auto) {
    return static_cast<Policy &>(seg);
  }

  friend auto unwrap(curve &&seg) -> decltype(auto) {
    return static_cast<Policy &&>(seg);
  }

  template <typename T> friend auto wrap_like(const curve &, T &&t) {
    return curve<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(curve &, T &&t) {
    return curve<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(curve &&, T &&t) {
    return curve<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }
};

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(const tf::curve<Dims, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(tf::curve<Dims, Policy> &t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t I, std::size_t Dims, typename Policy,
          typename = std::enable_if_t<
              tf::static_size_v<Policy> != tf::dynamic_size, void>>
auto get(tf::curve<Dims, Policy> &&t) -> decltype(auto) {
  using std::get;
  return get<I>(static_cast<const Policy &>(t));
}

template <std::size_t Dims, typename Policy>
struct static_size<tf::curve<Dims, Policy>> : static_size<Policy> {};

template <std::size_t V, typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_curve<V>(static_cast<Range0 &&>(ids),
                                       static_cast<Range1 &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

template <std::size_t V, typename Range> auto make_curve(Range &&points) {
  auto policy = tf::core::make_curve<V>(static_cast<Range &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

template <typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_curve(static_cast<Range0 &&>(ids),
                                    static_cast<Range1 &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

template <typename Range> auto make_curve(Range &&points) {
  auto policy = tf::core::make_curve(static_cast<Range &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

} // namespace tf
namespace std {
template <std::size_t Dims, typename Policy>
struct tuple_size<tf::curve<Dims, Policy>> : tuple_size<Policy> {};

template <std::size_t I, std::size_t Dims, typename Policy>
struct tuple_element<I, tf::curve<Dims, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<tf::curve<Dims, Policy>>().begin())>::value_type;
};

template <std::size_t I, typename Policy>
struct tuple_element<I, tf::curve<tf::dynamic_size, Policy>>;

} // namespace std

