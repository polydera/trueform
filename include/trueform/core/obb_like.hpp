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
#include "./base/obb_impl.hpp"
#include "./base/pt.hpp"
#include "./base/vec.hpp"
#include "./point.hpp"
#include "./unit_vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Base template for oriented bounding box types.
///
/// Provides the common interface for OBBs, defined by an origin point,
/// orthonormal axes, and extents along each axis. Includes a method for
/// computing the center.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
template <std::size_t Dims, typename Policy> struct obb_like : Policy {
  obb_like() = default;
  obb_like(const Policy &policy) : Policy{policy} {}
  obb_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::axes;
  using Policy::extent;
  using Policy::origin;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  /// @brief Compute the center point of the OBB.
  ///
  /// The origin is the local (0,0,0) corner, so center is origin + axes *
  /// (extent * 0.5).
  ///
  /// @return A `point<T, N>` representing the center.
  auto center() const -> point<coordinate_type, Dims> {
    auto out = origin;
    auto c = out.as_vector_view();
    for (std::size_t i = 0; i < Dims; ++i)
      c = c + axes[i] * (extent[i] * coordinate_type(0.5));
    return out;
  }

  /// @brief Return the dimensionality of the OBB.
  ///
  /// Provided as a compile-time constant.
  ///
  /// @return The number of spatial dimensions (N).
  constexpr auto size() -> std::size_t { return Dims; }

  template <typename RealT>
  operator tf::obb_like<Dims, tf::core::obb<Dims, tf::core::pt<RealT, Dims>,
                                            tf::core::vec<RealT, Dims>>>()
      const {
    std::array<tf::unit_vector<RealT, Dims>, Dims> converted_axes;
    for (std::size_t i = 0; i < Dims; ++i)
      converted_axes[i] = tf::make_unit_vector(tf::unsafe, axes[i]);
    std::array<RealT, Dims> converted_extent;
    for (std::size_t i = 0; i < Dims; ++i)
      converted_extent[i] = static_cast<RealT>(extent[i]);
    return {origin, converted_axes, converted_extent};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const obb_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<const Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(obb_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(obb_like<V, Policy> &&obj) -> decltype(auto) {
  return static_cast<Policy &&>(obj);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const obb_like<V, Policy> &, T &&t) {
  return obb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(obb_like<V, Policy> &, T &&t) {
  return obb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(obb_like<V, Policy> &&, T &&t) {
  return obb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const obb_like<V, Policy> &&, T &&t) {
  return obb_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_primitives
/// @brief Create an OBB view from point and vector views.
///
/// Returns a view when inputs are views, preserving zero-copy semantics.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The origin point policy.
/// @tparam Policy1 The axes unit vector policy.
/// @param origin The corner point of the OBB.
/// @param axes The orthonormal axes of the OBB.
/// @param extent The extents along each axis.
/// @return A @ref tf::obb_like instance.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_obb_like(
    const point_like<Dims, Policy0> &origin,
    const std::array<unit_vector_like<Dims, Policy1>, Dims> &axes,
    const std::array<tf::coordinate_type<Policy0>, Dims> &extent) {
  return tf::obb_like<Dims, tf::core::obb<Dims, Policy0, Policy1>>{origin, axes,
                                                                   extent};
}

} // namespace tf
