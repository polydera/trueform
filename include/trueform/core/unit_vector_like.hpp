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
#include "./base/normalize.hpp"
#include "./base/vec.hpp"
#include "./unsafe.hpp"
#include "./vector_like.hpp"
#include <type_traits>

namespace tf {

/// @ingroup core_primitives
/// @brief A fixed-size unit vector wrapper type.
///
/// `unit_vector_like<N, T>` represents a vector of dimension `N` with a fixed
/// length of 1. It inherits from `tf::vector_like<N, T>` but provides
/// guarantees and overrides to reflect the unit-length invariant. All instances
/// must be normalized at construction time.
///
/// Use `make_unit_vector()` to create
/// instances.
///
/// @tparam T The scalar type (e.g. float, double).
/// @tparam N The number of dimensions (e.g. 2, 3, 4).
template <std::size_t Dims, typename T>
struct unit_vector_like : tf::vector_like<Dims, T> {
private:
  using base_t = tf::vector_like<Dims, T>;
  template <typename U>
  using value_vec_type = unit_vector_like<Dims, core::vec<U, Dims>>;
  template <typename U>
  using value_base_type = tf::vector_like<Dims, core::vec<U, Dims>>;

  using self_value_type = value_vec_type<typename base_t::value_type>;

public:
  unit_vector_like() = default;
  unit_vector_like(const unit_vector_like &other) : base_t{other} {}
  unit_vector_like(unit_vector_like &&other) : base_t{other} {}
  unit_vector_like(const base_t &other) : base_t{other} { tf::core::normalize(*this); }
  unit_vector_like(base_t &&other) : base_t{other} { tf::core::normalize(*this); }
  unit_vector_like(tf::unsafe_t, const base_t &other) : base_t{other} {}
  unit_vector_like(tf::unsafe_t, base_t &&other) : base_t{other} {}

  template <typename U> operator value_vec_type<U>() const { return as<U>(); }

  template <typename U> auto as() const -> value_vec_type<U> {
    return value_vec_type<U>{tf::unsafe,
                             static_cast<value_base_type<U>>(*this)};
  }

  auto operator=(const unit_vector_like &v) -> unit_vector_like & {
    base_t::operator=(v);
    return *this;
  }
  auto operator=(unit_vector_like &v) -> unit_vector_like & {
    base_t::operator=(v);
    return *this;
  }

  template <typename U>
  auto operator=(const unit_vector_like<Dims, U> &v) -> unit_vector_like & {
    base_t::operator=(v);
    return *this;
  }
  template <typename U>
  auto operator=(unit_vector_like<Dims, U> &&v) -> unit_vector_like & {
    base_t::operator=(v);
    return *this;
  }

  auto operator=(const tf::vector_like<Dims, T> &)
      -> unit_vector_like & = delete;
  auto operator=(tf::vector_like<Dims, T> &&) -> unit_vector_like & = delete;

  friend auto operator-(const unit_vector_like &a) -> self_value_type {
    return self_value_type{tf::unsafe, -static_cast<const base_t &>(a)};
  }

  /// @brief Returns the squared length (always 1).
  /// @return The value `1`.
  constexpr auto length2() const -> T { return 1; }

  /// @brief Returns the length (always 1).
  /// @return The value `1`.
  constexpr auto length() const -> T { return 1; }

  // forbid asignment operator from base
  template <typename U>
  friend auto operator+=(unit_vector_like &a, const U &b)
      -> unit_vector_like & = delete;
  template <typename U>
  friend auto operator-=(unit_vector_like &a, const U &b)
      -> unit_vector_like & = delete;
  template <typename U>
  friend auto operator*=(unit_vector_like &a, const U &b)
      -> unit_vector_like & = delete;
  template <typename U>
  friend auto operator/=(unit_vector_like &a, const U &b)
      -> unit_vector_like & = delete;
};

template <std::size_t Dims, typename Policy>
struct static_size<unit_vector_like<Dims, Policy>>
    : std::integral_constant<std::size_t, Dims> {};

template <std::size_t Dims, typename Policy>
auto unwrap(const unit_vector_like<Dims, Policy> &vec) -> decltype(auto) {
  return static_cast<const Policy &>(vec);
}

template <std::size_t Dims, typename Policy>
auto unwrap(unit_vector_like<Dims, Policy> &vec) -> decltype(auto) {
  return static_cast<Policy &>(vec);
}

template <std::size_t Dims, typename Policy>
auto unwrap(unit_vector_like<Dims, Policy> &&vec) -> decltype(auto) {
  return static_cast<Policy &&>(vec);
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(const unit_vector_like<Dims, Policy> &, T &&t) {
  return unit_vector_like<Dims, std::decay_t<T>>{
      tf::unsafe, tf::vector_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)}};
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(unit_vector_like<Dims, Policy> &, T &&t) {
  return unit_vector_like<Dims, std::decay_t<T>>{
      tf::unsafe, tf::vector_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)}};
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(unit_vector_like<Dims, Policy> &&, T &&t) {
  return unit_vector_like<Dims, std::decay_t<T>>{
      tf::unsafe, tf::vector_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)}};
}

} // namespace tf

namespace std {

template <std::size_t I, std::size_t Dims, typename Policy>
struct tuple_element<I, tf::unit_vector_like<Dims, Policy>> {
  using type = typename tf::unit_vector_like<Dims, Policy>::value_type;
};

template <std::size_t Dims, typename Policy>
struct tuple_size<tf::unit_vector_like<Dims, Policy>>
    : std::integral_constant<std::size_t, Dims> {};

} // namespace std

namespace tf {

template <std::size_t I, std::size_t Dims, typename Policy>
constexpr auto get(const tf::unit_vector_like<Dims, Policy> &v) ->
    typename tf::unit_vector_like<Dims, Policy>::const_reference {
  static_assert(I < Dims);
  return v[I];
}

template <std::size_t I, std::size_t Dims, typename Policy>
constexpr auto get(tf::unit_vector_like<Dims, Policy> &v) ->
    typename tf::unit_vector_like<Dims, Policy>::reference {
  static_assert(I < Dims);
  return v[I];
}
} // namespace tf
