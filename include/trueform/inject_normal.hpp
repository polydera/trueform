/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once
#include "./unit_vector.hpp"
#include "./unit_vector_like.hpp"
#include "./unwrap.hpp"
#include <utility>

namespace tf {
/**
 * @ingroup injectors
 * @brief Type injector that adds a unit normal vector to an existing base
 * class.
 *
 * This struct composes a new type by inheriting from the given `Base` type and
 * injecting a `unit_vector<T, Dims>` representing the surface normal.
 *
 * It provides accessors for the injected normal and forwards the rest of the
 * interface to the base class. Designed to be used with the `inject_normal()`
 * helper.
 *
 * @tparam T    Underlying scalar type of the normal vector.
 * @tparam Dims Dimensionality of the normal vector (e.g., 2 or 3).
 * @tparam Base The type being augmented.
 */
template <typename T, std::size_t Dims, typename Base>
struct inject_normal_t : Base {
  /**
   * @brief Constructs an instance by copying the normal and the base.
   *
   * @param _normal The normal vector to inject.
   * @param base    The base object to extend.
   */
  inject_normal_t(const unit_vector<T, Dims> &_normal, const Base &base)
      : Base{base}, _normal{_normal} {}

  /**
   * @brief Constructs an instance by moving the normal and the base.
   *
   * @param _normal The normal vector to inject (moved).
   * @param base    The base object to extend (moved).
   */
  inject_normal_t(unit_vector<T, Dims> &&_normal, Base &&base)
      : Base{std::move(base)}, _normal{std::move(_normal)} {}

  /**
   * @brief Returns a const reference to the injected normal.
   */
  auto normal() const -> const unit_vector<T, Dims> & { return _normal; }

  /**
   * @brief Returns a mutable reference to the injected normal.
   */
  auto normal() -> unit_vector<T, Dims> & { return _normal; }

private:
  unit_vector<T, Dims> _normal;
};

namespace implementation {

template <typename RealT, std::size_t Dims, typename Base>
auto has_injected_normal(const tf::inject_normal_t<RealT, Dims, Base> *)
    -> std::true_type;

auto has_injected_normal(const void *) -> std::false_type;
} // namespace implementation

template <typename T>
inline constexpr bool has_injected_normal =
    decltype(implementation::has_injected_normal(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

/**
 * @ingroup injectors
 * @brief Constructs an `inject_normal_t` by injecting a normal into a base
 * type.
 *
 * This helper function provides a convenient way to inject a unit normal into
 * an object, returning a composed type with added `.normal()` accessors.
 *
 * @tparam T    Underlying scalar type of the normal vector.
 * @tparam Dims Dimensionality of the normal vector.
 * @tparam Base Type of the base object (will be decayed).
 * @param normal The normal vector to inject.
 * @param base   The base object to augment.
 * @return A composed object with normal support.
 */
template <std::size_t Dims, typename T, typename Base>
auto inject_normal(const unit_vector_like<Dims, T> &normal, Base &&base) {
  if constexpr (has_injected_normal<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, inject_normal_t<tf::value_type<T>, Dims,
                              std::decay_t<decltype(b_base)>>{normal, b_base});
  }
}

} // namespace tf
