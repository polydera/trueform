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
#include "./static_size.hpp"
#include <cstddef>
namespace tf::core {
namespace detail {
template <std::size_t Size, typename Policy>
struct non_assignable_range : Policy {
  using Policy::Policy;
  non_assignable_range(const Policy &p) : Policy{p} {}
  non_assignable_range(Policy &&p) : Policy{std::move(p)} {}
  non_assignable_range() = default;
  non_assignable_range(const non_assignable_range &) = default;
  non_assignable_range(non_assignable_range &&) = default;
  auto operator=(const non_assignable_range &)
      -> non_assignable_range & = delete;
  auto operator=(non_assignable_range &&) -> non_assignable_range & = delete;
};
template <std::size_t Size, typename Policy> struct assignable_range : Policy {
  using Policy::Policy;
  assignable_range(const Policy &p) : Policy{p} {}
  assignable_range(Policy &&p) : Policy{std::move(p)} {}
  assignable_range() = default;
  assignable_range(const assignable_range &) = default;
  assignable_range(assignable_range &&) = default;
  auto operator=(const assignable_range &other) -> assignable_range & {
    for (std::size_t i = 0; i < Policy::size(); ++i)
      Policy::operator[](i) = other[i];
    return *this;
  }

  auto operator=(assignable_range &&other) -> assignable_range & {
    for (std::size_t i = 0; i < Policy::size(); ++i)
      Policy::operator[](i) = other[i];
    return *this;
  }

  template <typename Policy1>
  auto
  operator=(const assignable_range<Size, Policy1> &other) -> std::enable_if_t<
      std::is_assignable_v<decltype(Policy::operator[](0)), decltype(other[0])>,
      assignable_range &> {
    for (std::size_t i = 0; i < Policy::size(); ++i)
      Policy::operator[](i) = other[i];
    return *this;
  }

  template <typename Policy1>
  auto operator=(const non_assignable_range<Size, Policy1> &other)
      -> std::enable_if_t<std::is_assignable_v<decltype(Policy::operator[](0)),
                                               decltype(other[0])>,
                          assignable_range &> {
    for (std::size_t i = 0; i < Policy::size(); ++i)
      Policy::operator[](i) = other[i];
    return *this;
  }

  template <typename T>
  auto operator=(const std::array<T, Size> &other) -> std::enable_if_t<
      (Size != tf::dynamic_size) &&
          std::is_assignable_v<decltype(Policy::operator[](0)),
                               decltype(other[0])>,
      assignable_range &> {
    for (std::size_t i = 0; i < Policy::size(); ++i)
      Policy::operator[](i) = other[i];
    return *this;
  }
};
} // namespace detail
template <std::size_t Size, typename Policy>
using assignable_range =
    std::conditional_t<std::is_assignable_v<typename Policy::reference,
                                            typename Policy::reference>,
                       detail::assignable_range<Size, Policy>,
                       detail::non_assignable_range<Size, Policy>>;
} // namespace tf::core
namespace tf {
template <std::size_t Size, typename Policy>
struct static_size<tf::core::detail::assignable_range<Size, Policy>>
    : std::integral_constant<std::size_t, Size> {};
template <std::size_t Size, typename Policy>
struct static_size<tf::core::detail::non_assignable_range<Size, Policy>>
    : std::integral_constant<std::size_t, Size> {};
} // namespace tf
namespace std {
template <std::size_t Size, typename Policy>
struct tuple_size<tf::core::detail::assignable_range<Size, Policy>>
    : tuple_size<Policy> {};
template <std::size_t I, std::size_t Size, typename Policy>
struct tuple_element<I, tf::core::detail::assignable_range<Size, Policy>>
    : tuple_element<I, Policy> {};
template <std::size_t Size, typename Policy>
struct tuple_size<tf::core::detail::non_assignable_range<Size, Policy>>
    : tuple_size<Policy> {};
template <std::size_t I, std::size_t Size, typename Policy>
struct tuple_element<I, tf::core::detail::non_assignable_range<Size, Policy>>
    : tuple_element<I, Policy> {};
} // namespace std
