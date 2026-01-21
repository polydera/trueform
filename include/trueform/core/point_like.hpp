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
#include "./base/pt.hpp"
#include "./sqrt.hpp"
#include "./static_size.hpp"
#include "./vector_like.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup core_primitives
/// @brief A general-purpose point interface for fixed-size spatial points.
///
/// @tparam Dims The number of dimensions (e.g. 2, 3, 4).
/// @tparam Policy A type that provides access to the underlying data buffer and
/// defines `coordinate_type`.
template <std::size_t Dims, typename Policy> struct point_like : public Policy {
private:
  using original_type = std::remove_reference_t<typename Policy::element_type>;
  using clean_element_type =
      std::remove_cv_t<std::remove_reference_t<typename Policy::element_type>>;
  template <typename B>
  using vec_common_element_t = std::common_type_t<
      clean_element_type,
      std::remove_cv_t<std::remove_reference_t<typename B::element_type>>>;

  template <typename B>
  using pt_vec_result_t =
      point_like<Dims, core::pt<vec_common_element_t<B>, Dims>>;

  template <typename B>
  using pt_pt_result_t =
      vector_like<Dims, core::vec<vec_common_element_t<B>, Dims>>;

  template <typename U>
  using value_pt_type = point_like<Dims, core::pt<U, Dims>>;

  template <typename U>
  using value_vec_type = vector_like<Dims, core::vec<U, Dims>>;

  using vector_view_type =
      vector_like<Dims, core::vec_view<original_type, Dims>>;
  using const_vector_view_type =
      vector_like<Dims, core::vec_view<const original_type, Dims>>;

public:
  using value_type = clean_element_type;
  using reference = typename Policy::element_type &;
  using const_reference = const typename Policy::element_type &;
  using pointer = typename Policy::element_type *;
  using const_pointer = const typename Policy::element_type *;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using Policy::Policy;
  using Policy::operator=;
  point_like() = default;
  point_like(const point_like &) = default;
  point_like(point_like &&) = default;
  auto operator=(const point_like &) -> point_like & = default;
  auto operator=(point_like &&) -> point_like & = default;
  point_like(const Policy &policy) : Policy{policy} {}
  point_like(Policy &&policy) : Policy{std::move(policy)} {}

  /// @brief Returns pointer to the internal data.
  auto data() -> pointer { return Policy::data(); }

  /// @brief Returns const pointer to the internal data.
  auto data() const -> const_pointer { return Policy::data(); }

  /// @brief Access element at index `i` (const).
  auto operator[](std::size_t i) const -> const_reference { return data()[i]; }

  /// @brief Access element at index `i` (mutable).
  auto operator[](std::size_t i) -> reference { return data()[i]; }

  /// @brief Iterator to beginning (mutable).
  auto begin() -> pointer { return data(); }

  /// @brief Iterator to beginning (const).
  auto begin() const -> const_pointer { return data(); }

  /// @brief Iterator to end (mutable).
  auto end() -> pointer { return data() + Dims; }

  /// @brief Iterator to end (const).
  auto end() const -> const_pointer { return data() + Dims; }

  /// @brief Returns the number of dimensions.
  static constexpr std::size_t size() { return Dims; }

  /// @brief Returns squared length
  auto length2() const -> value_type {
    value_type out{0};
    for (std::size_t i = 0; i < Dims; ++i)
      out += data()[i] * data()[i];
    return out;
  }

  /// @brief Returns length
  auto length() const -> value_type { return tf::sqrt(length2()); }

  /// @brief Converts this vector to another value type `U`.
  /// @tparam U The target scalar type.
  /// @return A new point_like instance with elements converted to `U`.
  template <typename U> auto as() const -> value_pt_type<U> {
    value_pt_type<U> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = Policy::data()[i];
    return out;
  }

  auto as_vector() const -> value_vec_type<value_type> {
    value_vec_type<value_type> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = Policy::data()[i];
    return out;
  }

  auto as_vector_view() const -> const_vector_view_type {
    return const_vector_view_type{begin()};
  }

  auto as_vector_view() -> vector_view_type {
    return vector_view_type{begin()};
  }

  /// @brief Implicit conversion to vector of another type.
  template <typename U> operator value_pt_type<U>() const { return as<U>(); }

  /// @brief Adds vector `b` to `a` element-wise.
  template <typename B>
  friend auto
  operator+=(point_like &a, const vector_like<Dims, B> &b) -> std::enable_if_t<
      std::is_assignable_v<reference, typename B::element_type>, point_like &> {
    for (std::size_t i = 0; i < Dims; ++i)
      a[i] += b[i];
    return a;
  }

  template <typename B>
  friend auto operator+=(point_like &&a, const vector_like<Dims, B> &b)
      -> std::enable_if_t<
          std::is_assignable_v<reference, typename B::element_type>,
          point_like &&> {
    for (std::size_t i = 0; i < Dims; ++i)
      a[i] += b[i];
    return static_cast<point_like &&>(a);
  }

  /// @brief Subtracts vector `b` from `a` element-wise.
  template <typename B>
  friend auto operator-=(point_like &a, const vector_like<Dims, B> &b)
      -> std::enable_if_t<
          std::is_assignable_v<reference &, typename B::element_type>,
          point_like &> {
    for (std::size_t i = 0; i < Dims; ++i)
      a[i] -= b[i];
    return a;
  }

  /// @brief Subtracts vector `b` from `a` element-wise.
  template <typename B>
  friend auto operator-=(point_like &&a, const vector_like<Dims, B> &b)
      -> std::enable_if_t<
          std::is_assignable_v<reference &, typename B::element_type>,
          point_like &&> {
    for (std::size_t i = 0; i < Dims; ++i)
      a[i] -= b[i];
    return static_cast<point_like &&>(a);
  }

  /// @brief Element-wise addition of vectors `a` and `b`.
  template <typename B>
  friend auto operator+(const point_like &a, const vector_like<Dims, B> &b)
      -> pt_vec_result_t<B> {
    pt_vec_result_t<B> out = a;
    out += b;
    return out;
  }

  /// @brief Element-wise addition of vectors `a` and `b`.
  template <typename B>
  friend auto operator+(const vector_like<Dims, B> &a, const point_like &b)
      -> pt_vec_result_t<B> {
    return b + a;
  }

  /// @brief Element-wise addition of vectors `a` and `b`.
  template <typename B>
  friend auto operator-(const point_like &a, const vector_like<Dims, B> &b)
      -> pt_vec_result_t<B> {
    pt_vec_result_t<B> out = a;
    out -= b;
    return out;
  }

  /// @brief Element-wise subtraction of vectors `a` and `b`.
  template <typename B>
  friend auto operator-(const point_like &a, const point_like<Dims, B> &b)
      -> pt_pt_result_t<B> {
    pt_pt_result_t<B> out;
    for (std::size_t i = 0; i < Dims; ++i)
      out[i] = a[i] - b[i];
    return out;
  }

  /// @brief Equality comparison (element-wise).
  template <typename B>
  friend auto operator==(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    for (std::size_t i = 0; i < Dims; ++i)
      if (a[i] != b[i])
        return false;
    return true;
  }

  /// @brief Inequality comparison.
  template <typename B>
  friend auto operator!=(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    return !(a == b);
  }

  /// @brief Lexicographical comparison.
  template <typename B>
  friend auto operator<(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    for (std::size_t i = 0; i < Dims; ++i) {
      if (a[i] < b[i])
        return true;
      if (a[i] > b[i])
        return false;
    }
    return false;
  }

  /// @brief Lexicographical greater-than comparison.
  template <typename B>
  friend auto operator>(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    return b < a;
  }

  /// @brief Lexicographical less-than-or-equal comparison.
  template <typename B>
  friend auto operator<=(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    return !(b < a);
  }

  /// @brief Lexicographical greater-than-or-equal comparison.
  template <typename B>
  friend auto operator>=(const point_like &a, const point_like<Dims, B> &b)
      -> bool {
    return !(a < b);
  }

  friend auto swap(point_like &a, point_like &b) -> void {
    for (std::size_t i = 0; i < Dims; ++i)
      std::swap(a[i], b[i]);
  }
};

template <std::size_t Dims, typename Policy>
struct static_size<point_like<Dims, Policy>>
    : std::integral_constant<std::size_t, Dims> {};

template <std::size_t Dims, typename Policy>
auto unwrap(const point_like<Dims, Policy> &vec) -> decltype(auto) {
  return static_cast<const Policy &>(vec);
}

template <std::size_t Dims, typename Policy>
auto unwrap(point_like<Dims, Policy> &vec) -> decltype(auto) {
  return static_cast<Policy &>(vec);
}

template <std::size_t Dims, typename Policy>
auto unwrap(point_like<Dims, Policy> &&vec) -> decltype(auto) {
  return static_cast<Policy &&>(vec);
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(const point_like<Dims, Policy> &, T &&t) {
  return point_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(point_like<Dims, Policy> &, T &&t) {
  return point_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy, typename T>
auto wrap_like(point_like<Dims, Policy> &&, T &&t) {
  return point_like<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
}

} // namespace tf

namespace std {

template <std::size_t I, std::size_t Dims, typename Policy>
struct tuple_element<I, tf::point_like<Dims, Policy>> {
  using type = typename tf::point_like<Dims, Policy>::value_type;
};

template <std::size_t Dims, typename Policy>
struct tuple_size<tf::point_like<Dims, Policy>>
    : std::integral_constant<std::size_t, Dims> {};

} // namespace std

namespace tf {

template <std::size_t I, std::size_t Dims, typename Policy>
constexpr auto get(const tf::point_like<Dims, Policy> &v) ->
    typename tf::point_like<Dims, Policy>::const_reference {
  static_assert(I < Dims);
  return v[I];
}

template <std::size_t I, std::size_t Dims, typename Policy>
constexpr auto get(tf::point_like<Dims, Policy> &v) ->
    typename tf::point_like<Dims, Policy>::reference {
  static_assert(I < Dims);
  return v[I];
}

} // namespace tf
