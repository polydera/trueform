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
#include "./assignable_range.hpp"
namespace tf {
template <std::size_t Size, typename Policy>
class array_like : public tf::core::assignable_range<Size, Policy> {
  using base_t = tf::core::assignable_range<Size, Policy>;

public:
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;
  using base_t::operator=;
  array_like(const array_like &) = default;
  array_like(array_like &&) = default;
  auto operator=(const array_like &) -> array_like & = default;
  auto operator=(array_like &&) -> array_like & = default;

  explicit array_like(const Policy &p) : base_t(p) {}
  explicit array_like(Policy &&p) noexcept(
      std::is_nothrow_move_constructible<Policy>::value)
      : base_t(std::move(p)) {}

  template <typename T> operator array_like<Size, std::array<T, Size>>() const {
    std::array<T, Size> out;
    for (std::size_t i = 0; i < Size; ++i)
      out[i] = (*this)[i];
    return array_like<Size, std::array<T, Size>>{out};
  }

  template <typename Policy1>
  friend auto swap(array_like &lhs, array_like<Size, Policy1> &rhs)
      -> std::enable_if_t<std::is_swappable_with_v<typename Policy::reference,
                                                   typename Policy1::reference>,
                          void> {
    using std::swap;
    for (std::size_t i = 0; i < Size; ++i)
      swap(lhs[i], rhs[i]);
  }

  template <typename Policy1>
  friend auto swap(array_like &&lhs, array_like<Size, Policy1> &&rhs)
      -> std::enable_if_t<std::is_swappable_with_v<typename Policy::reference,
                                                   typename Policy1::reference>,
                          void> {
    using std::swap;
    for (std::size_t i = 0; i < Size; ++i)
      swap(lhs[i], rhs[i]);
  }

  template <typename Policy1>
  friend auto operator==(const array_like &lhs,
                         const array_like<Size, Policy1> &rhs) -> bool {
    for (std::size_t i = 0; i < Size; ++i)
      if (!(lhs[i] == rhs[i]))
        return false;
    return true;
  }

  template <typename Policy1>
  friend auto operator!=(const array_like &lhs,
                         const array_like<Size, Policy1> &rhs) -> bool {
    return !(lhs == rhs);
  }

  template <typename Policy1>
  friend auto operator<(const array_like &lhs,
                        const array_like<Size, Policy1> &rhs) -> bool {
    for (std::size_t i = 0; i < Size; ++i) {
      if (lhs[i] < rhs[i])
        return true;
      if (rhs[i] < lhs[i])
        return false;
    }
    return false;
  }

  template <typename Policy1>
  friend auto operator>(const array_like &lhs,
                        const array_like<Size, Policy1> &rhs) -> bool {
    return rhs < lhs;
  }

  template <typename Policy1>
  friend auto operator<=(const array_like &lhs,
                         const array_like<Size, Policy1> &rhs) -> bool {
    return !(rhs < lhs);
  }

  template <typename Policy1>
  friend auto operator>=(const array_like &lhs,
                         const array_like<Size, Policy1> &rhs) -> bool {
    return !(lhs < rhs);
  }
};

template <std::size_t Size, typename Policy>
struct static_size<array_like<Size, Policy>>
    : std::integral_constant<std::size_t, Size> {};

// get<I> overloads
template <std::size_t I, std::size_t Size, typename Policy>
auto get(const array_like<Size, Policy> &a) -> decltype(auto) {
  static_assert(Size != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < Size, "Index out of bounds");
  return a[I];
}

template <std::size_t I, std::size_t Size, typename Policy>
auto get(array_like<Size, Policy> &a) -> decltype(auto) {
  static_assert(Size != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < Size, "Index out of bounds");
  return a[I];
}

template <std::size_t I, std::size_t Size, typename Policy>
auto get(array_like<Size, Policy> &&a) -> decltype(auto) {
  static_assert(Size != tf::dynamic_size, "Dynamic size does not support get.");
  static_assert(I < Size, "Index out of bounds");
  return a[I];
}

template <std::size_t Size, typename Policy>
auto make_array_like(array_like<Size, Policy> array) {
  return array;
}

template <std::size_t Size, typename Policy>
auto make_array_like(Policy &&policy) {
  return array_like<Size, std::decay_t<Policy>>{static_cast<Policy &&>(policy)};
}

template <typename Policy> auto make_array_like(Policy &&policy) {
  return make_array_like<tf::static_size_v<Policy>>(
      static_cast<Policy &&>(policy));
}
} // namespace tf

namespace std {

template <std::size_t Size, typename Policy>
struct tuple_size<tf::array_like<Size, Policy>>
    : std::integral_constant<std::size_t, Size> {};

template <std::size_t I, std::size_t Size, typename Policy>
struct tuple_element<I, tf::array_like<Size, Policy>> {
  using type = typename Policy::value_type;
};

} // namespace std
