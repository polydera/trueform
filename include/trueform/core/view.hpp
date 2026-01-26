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

#include "./range.hpp"

namespace tf {

template <typename Policy> struct view : Policy {
  view(const Policy &r) : Policy{r} {}
  view(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const view<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(view<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(view<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const view<Policy> &, T &&t) {
  return view<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(view<Policy> &, T &&t) {
  return view<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(view<Policy> &&, T &&t) {
  return view<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_view(Range &&range) {
  auto r = tf::make_range<tf::static_size_v<Range>>(range.begin(), range.end());
  return view<decltype(r)>{std::move(r)};
}

template <std::size_t Size, typename Range> auto make_view(Range &&range) {
  auto r = tf::make_range<Size>(range.begin(), range.end());
  return view<decltype(r)>{std::move(r)};
}

template <typename Range> auto make_view(view<Range> r) -> view<Range> {
  return r;
}
} // namespace tf
namespace std {
template <typename Policy>
struct tuple_size<tf::view<Policy>> : tuple_size<Policy> {};

template <std::size_t I, typename Policy>
struct tuple_element<I, tf::view<Policy>> : tuple_element<I, Policy> {};
} // namespace std
