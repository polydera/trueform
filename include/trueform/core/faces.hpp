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
/// @brief Semantic wrapper marking a range as face connectivity.
///
/// Wraps any range to indicate it represents face indices.
/// Used with @ref tf::polygons to distinguish face data from point data.
///
/// @tparam Policy The underlying range policy.
template <typename Policy> struct faces : Policy {
  faces(const Policy &r) : Policy{r} {}
  faces(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const faces<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(faces<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(faces<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const faces<Policy> &, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(faces<Policy> &, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(faces<Policy> &&, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create a faces wrapper from a range.
///
/// @tparam Range The input range type.
/// @param r The range of face data.
/// @return A @ref tf::faces wrapping the range.
template <typename Range> auto make_faces(Range &&r) {
  auto r0 = tf::make_range(r);
  return tf::faces<decltype(r0)>{r0};
}

template <typename Range> auto make_faces(faces<Range> r) -> faces<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::faces<Policy> &obj) {
  return obj;
}
} // namespace tf
