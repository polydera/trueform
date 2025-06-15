/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "./implementation/inject_plane.hpp"
#include "./plane.hpp"
#include "./unwrap.hpp"

namespace tf {
/**
 * @ingroup injectors
 * @brief Type injector that adds a geometric plane to an existing base class.
 *
 * This struct composes a new type by inheriting from the given `Base` type
 * and injecting a `tf::plane<T, Dims>` member. It exposes accessors for the
 * plane itself as well as its normal vector.
 *
 * Useful when attaching planar geometric context to entities during
 * construction, while preserving their original interface via inheritance from
 * `Base`.
 *
 * @tparam T    Underlying scalar type of the plane.
 * @tparam Dims Dimensionality (usually 2 or 3).
 * @tparam Base The type being augmented.
 */
template <typename T, std::size_t Dims, typename Base>
struct inject_plane_t : tf::implementation::inject_plane_t<T, Dims, Base> {
private:
  using base_t = tf::implementation::inject_plane_t<T, Dims, Base>;

public:
  using base_t::base_t;
};

namespace implementation {

template <typename RealT, std::size_t Dims, typename Base>
auto has_injected_plane(const tf::inject_plane_t<RealT, Dims, Base> *)
    -> std::true_type;

auto has_injected_plane(const void *) -> std::false_type;
} // namespace implementation

template <typename T>
inline constexpr bool has_injected_plane =
    decltype(implementation::has_injected_plane(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

/**
 * @ingroup injectors
 * @brief Constructs an `inject_plane_t` by injecting a plane into a base type.
 *
 * This helper function provides a convenient way to inject a `tf::plane`
 * into an object, returning a composed type with `.plane()` and `.normal()`
 * accessors.
 *
 * @tparam T    Underlying scalar type of the plane.
 * @tparam Dims Dimensionality of the plane.
 * @tparam Base Type of the base object (will be decayed).
 * @param plane The plane to inject.
 * @param base  The base object to augment.
 * @return A composed object with plane support.
 */
template <typename T, std::size_t Dims, typename Base>
auto inject_plane(const plane<T, Dims> &plane, Base &&base) -> decltype(auto) {
  if constexpr (has_injected_plane<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        inject_plane_t<T, Dims, std::decay_t<decltype(b_base)>>{plane, b_base});
  }
}

} // namespace tf
