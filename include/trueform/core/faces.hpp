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
#include "./views/offset_block_range.hpp"
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

/// @ingroup core_ranges
/// @brief Create a faces wrapper from flat indices with fixed polygon size.
///
/// @tparam Ngons The number of vertices per polygon (e.g., 3 for triangles).
/// @tparam Range The input range type.
/// @param flat_ids Flat array of vertex indices.
/// @return A @ref tf::faces wrapping a blocked range.
template <std::size_t Ngons, typename Range> auto make_faces(Range &&flat_ids) {
  return make_faces(tf::make_blocked_range<Ngons>(flat_ids));
}

/// @ingroup core_ranges
/// @brief Create a faces wrapper for variable-size polygons.
///
/// @tparam Range0 The offsets range type.
/// @tparam Range1 The data range type.
/// @param offsets Array of offsets defining polygon boundaries.
/// @param flat_ids Flat array of vertex indices.
/// @return A @ref tf::faces wrapping an offset block range.
template <typename Range0, typename Range1>
auto make_faces(Range0 &&offsets, Range1 &&flat_ids) {
  return make_faces(tf::make_offset_block_range(offsets, flat_ids));
}

template <typename Range> auto make_faces(faces<Range> r) -> faces<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::faces<Policy> &obj) {
  return obj;
}
} // namespace tf
