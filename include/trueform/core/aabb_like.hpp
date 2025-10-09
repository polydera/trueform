/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/aabb.hpp"
#include "./point.hpp"
#include "./vector.hpp"

namespace tf {
template <std::size_t Dims, typename Policy> struct aabb_like : Policy {
  aabb_like() = default;
  aabb_like(const Policy &policy) : Policy{policy} {}
  aabb_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::max;
  using Policy::min;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  /// @brief Compute the center point of the AABB.
  ///
  /// Returns the midpoint between `min` and `max`.
  ///
  /// @return A `vector<T, N>` representing the center.
  auto center() const -> point<coordinate_type, Dims> {
    return tf::make_point((min.as_vector_view() + max.as_vector_view()) *
                          static_cast<coordinate_type>(0.5));
  }

  /// @brief Compute the diagonal vector of the AABB.
  ///
  /// Returns the vector from `min` to `max`, representing the box’s size
  /// along each axis.
  ///
  /// @return A `vector<T, N>` representing the diagonal.
  auto diagonal() const -> vector<coordinate_type, Dims> { return max - min; }

  /// @brief Return the dimensionality of the AABB.
  ///
  /// Provided as a compile-time constant.
  ///
  /// @return The number of spatial dimensions (N).
  constexpr auto size() -> std::size_t { return Dims; }

  template <typename RealT>
  operator tf::aabb_like<Dims, tf::core::aabb<Dims, tf::core::pt<RealT, Dims>,
                                              tf::core::pt<RealT, Dims>>>()
      const {
    return {min, max};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const aabb_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(aabb_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(aabb_like<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const aabb_like<V, Policy> &, T &&t) {
  return aabb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(aabb_like<V, Policy> &, T &&t) {
  return aabb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(aabb_like<V, Policy> &&, T &&t) {
  return aabb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const aabb_like<V, Policy> &&, T &&t) {
  return aabb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_aabb_like(const point_like<Dims, Policy0> &origin,
                    const point_like<Dims, Policy1> &direction) {
  return tf::aabb_like<Dims, tf::core::aabb<Dims, Policy0, Policy1>>{origin,
                                                                     direction};
}

} // namespace tf
