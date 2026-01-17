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
#include "./views/blocked_range.hpp"
#include <type_traits>

namespace tf {

/// @ingroup core_ranges
/// @brief Semantic wrapper marking a range as edge connectivity.
///
/// Wraps any range to indicate it represents edge indices.
/// Used with @ref tf::segments to distinguish edge data from point data.
///
/// @tparam Policy The underlying range policy.
template <typename Policy> struct edges : Policy {
  edges(const Policy &r) : Policy{r} {}
  edges(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const edges<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(edges<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(edges<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const edges<Policy> &, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(edges<Policy> &, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(edges<Policy> &&, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create an edges wrapper from a range.
///
/// Automatically detects whether the input is a flat array of indices
/// or already structured as pairs. Flat arrays are blocked by 2.
///
/// @tparam Range The input range type.
/// @param r The range of edge data.
/// @return An @ref tf::edges wrapping the range.
template <typename Range> auto make_edges(Range &&r) {
  auto r0 = tf::make_range(r);
  if constexpr (tf::static_size_v<decltype(r0[0])> == 2) {
    // Already structured as pairs, just wrap
    return tf::edges<decltype(r0)>{r0};
  } else {
    // Flat array of scalars, block by 2
    return make_edges(tf::make_blocked_range<2>(r0));
  }
}

template <typename Range> auto make_edges(edges<Range> r) -> edges<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::edges<Policy> &obj) {
  return obj;
}
} // namespace tf
