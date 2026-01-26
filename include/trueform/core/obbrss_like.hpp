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
#include "./base/obbrss_impl.hpp"
#include "./base/pt.hpp"
#include "./base/vec.hpp"
#include "./point.hpp"
#include "./unit_vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Wrapper for combined OBB and RSS bounding volume.
///
/// Provides a unified interface for accessing both OBB and RSS properties
/// from a single bounding volume structure.
///
/// @tparam Dims The spatial dimension.
/// @tparam Policy The underlying storage policy.
template <std::size_t Dims, typename Policy> struct obbrss_like : Policy {
  obbrss_like() = default;
  obbrss_like(const Policy &policy) : Policy{policy} {}
  obbrss_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::axes;
  using Policy::extent;
  using Policy::length;
  using Policy::obb_origin;
  using Policy::rss_origin;
  using Policy::radius;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  /// @brief Compute the center point of the OBB.
  ///
  /// The obb_origin is the local corner, so center is obb_origin + axes *
  /// (extent * 0.5).
  ///
  /// @return A `point<T, N>` representing the OBB center.
  auto obb_center() const -> point<coordinate_type, Dims> {
    auto out = obb_origin;
    auto c = out.as_vector_view();
    for (std::size_t i = 0; i < Dims; ++i)
      c = c + axes[i] * (extent[i] * coordinate_type(0.5));
    return out;
  }

  /// @brief Compute the center point of the RSS rectangle.
  ///
  /// The rss_origin is the local corner, so center is
  /// rss_origin + axes[0] * (length[0] * 0.5) + ... + axes[Dims-2] *
  /// (length[Dims-2] * 0.5).
  ///
  /// @return A `point<T, N>` representing the RSS rectangle center.
  auto rss_center() const -> point<coordinate_type, Dims> {
    auto out = rss_origin;
    auto c = out.as_vector_view();
    for (std::size_t i = 0; i < Dims - 1; ++i)
      c = c + axes[i] * (length[i] * coordinate_type(0.5));
    return out;
  }

  /// @brief Return the dimensionality of the bounding volume.
  ///
  /// @return The number of spatial dimensions (N).
  constexpr auto size() -> std::size_t { return Dims; }

  template <typename RealT>
  operator tf::obbrss_like<
      Dims, tf::core::obbrss<Dims, tf::core::pt<RealT, Dims>,
                             tf::core::vec<RealT, Dims>>>() const {
    std::array<tf::unit_vector<RealT, Dims>, Dims> converted_axes;
    for (std::size_t i = 0; i < Dims; ++i)
      converted_axes[i] = tf::make_unit_vector(tf::unsafe, axes[i]);
    std::array<RealT, Dims> converted_extent;
    for (std::size_t i = 0; i < Dims; ++i)
      converted_extent[i] = static_cast<RealT>(extent[i]);
    std::array<RealT, Dims - 1> converted_length;
    for (std::size_t i = 0; i < Dims - 1; ++i)
      converted_length[i] = static_cast<RealT>(length[i]);
    return {obb_origin, rss_origin, converted_axes, converted_extent,
            converted_length, static_cast<RealT>(radius)};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const obbrss_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<const Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(obbrss_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(obbrss_like<V, Policy> &&obj) -> decltype(auto) {
  return static_cast<Policy &&>(obj);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const obbrss_like<V, Policy> &, T &&t) {
  return obbrss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(obbrss_like<V, Policy> &, T &&t) {
  return obbrss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(obbrss_like<V, Policy> &&, T &&t) {
  return obbrss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const obbrss_like<V, Policy> &&, T &&t) {
  return obbrss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_primitives
/// @brief Create an OBB-RSS hybrid bounding volume.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy0 The point policy type.
/// @tparam Policy1 The unit vector policy type.
/// @param obb_origin The OBB corner point.
/// @param rss_origin The RSS rectangle corner point.
/// @param axes The orthonormal basis axes.
/// @param extent The OBB half-extents along each axis.
/// @param length The RSS rectangle lengths.
/// @param radius The RSS capsule radius.
/// @return An @ref tf::obbrss_like bounding volume.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_obbrss_like(
    const point_like<Dims, Policy0> &obb_origin,
    const point_like<Dims, Policy0> &rss_origin,
    const std::array<unit_vector_like<Dims, Policy1>, Dims> &axes,
    const std::array<tf::coordinate_type<Policy0>, Dims> &extent,
    const std::array<tf::coordinate_type<Policy0>, Dims - 1> &length,
    tf::coordinate_type<Policy0> radius) {
  return tf::obbrss_like<Dims, tf::core::obbrss<Dims, Policy0, Policy1>>{
      obb_origin, rss_origin, axes, extent, length, radius};
}

} // namespace tf
