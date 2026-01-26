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
#include <type_traits>

namespace tf {

/// @ingroup core_ranges
/// @brief Semantic wrapper marking a range as path connectivity.
///
/// Wraps any range to indicate it represents path indices.
/// Used with @ref tf::curves to distinguish path data from point data.
///
/// @tparam Policy The underlying range policy.
template <typename Policy> struct paths : Policy {
  paths(const Policy &r) : Policy{r} {}
  paths(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const paths<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(paths<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(paths<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const paths<Policy> &, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(paths<Policy> &, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(paths<Policy> &&, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create a paths wrapper from a range.
///
/// @tparam Range The input range type.
/// @param r The range of path data.
/// @return A @ref tf::paths wrapping the range.
template <typename Range> auto make_paths(Range &&r) {
  auto r0 = tf::make_range(r);
  return tf::paths<decltype(r0)>{r0};
}

template <typename Range> auto make_paths(paths<Range> r) -> paths<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::paths<Policy> &obj) {
  return obj;
}
} // namespace tf

