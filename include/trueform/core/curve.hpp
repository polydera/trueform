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

#include "./base/curve.hpp"
#include "./static_size.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A single curve as a sequence of points.
///
/// Wraps a policy providing indexed access to curve points.
/// Supports structured bindings when the curve has static size.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy The underlying curve policy.
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

/// @ingroup core_primitives
/// @brief Create a curve from point indices and points.
///
/// @tparam V The static size (number of points in curve).
/// @tparam Range0 The index range type.
/// @tparam Range1 The points range type.
/// @param ids Point indices defining the curve path.
/// @param points The point data.
/// @return A @ref tf::curve.
template <std::size_t V, typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_curve<V>(static_cast<Range0 &&>(ids),
                                       static_cast<Range1 &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Create a curve from a sequence of points.
/// @overload
template <std::size_t V, typename Range> auto make_curve(Range &&points) {
  auto policy = tf::core::make_curve<V>(static_cast<Range &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Create a curve with dynamic size.
/// @overload
template <typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_curve(static_cast<Range0 &&>(ids),
                                    static_cast<Range1 &&>(points));
  return tf::curve<tf::static_size_v<decltype(points[0])>, decltype(policy)>(
      std::move(policy));
}

/// @ingroup core_primitives
/// @brief Create a curve with dynamic size from points.
/// @overload
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

