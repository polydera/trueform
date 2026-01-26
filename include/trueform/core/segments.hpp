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
#include "./base/segments.hpp"
#include "./base/soup.hpp"
#include "./edges.hpp"
#include "./form.hpp"
#include "./points.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief A range of line segments.
///
/// Wraps a range policy providing access to segment primitives.
/// Inherits from form to enable policy composition with spatial trees,
/// frames, and other policies.
///
/// @tparam Policy The underlying range policy.
template <typename Policy>
struct segments : form<coordinate_dims_v<Policy>, Policy> {
  using base = form<coordinate_dims_v<Policy>, Policy>;
  segments(const Policy &r) : base{r} {}
  segments(Policy &&r) : base{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const segments<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(segments<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(segments<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const segments<Policy> &, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segments<Policy> &, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segments<Policy> &&, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create a range of segments from edges and points.
///
/// @tparam Range0 The edge index range type.
/// @tparam Range1 The point range type.
/// @param edges A range of edge index pairs.
/// @param points A range of points.
/// @return A @ref tf::segments range.
template <typename Range0, typename Range1>
auto make_segments(Range0 &&edges, Range1 &&points) {
  auto r0 = tf::make_edges(edges);
  auto r1 = tf::make_points(points);
  return segments<core::segments<decltype(r0), decltype(r1)>>{
      core::segments<decltype(r0), decltype(r1)>{r0, r1}};
}

/// @ingroup core_ranges
/// @brief Identity overload for already-wrapped segment ranges.
template <typename Range>
auto make_segments(segments<Range> p) -> segments<Range> {
  return p;
}

/// @ingroup core_ranges
/// @brief Create a segment soup from a generic range.
///
/// Wraps a range of segment primitives into a @ref tf::segments range
/// using the soup adapter.
template <typename Range> auto make_segments(Range &&r) {
  auto segs = tf::make_range(r);
  return segments<core::soup<decltype(segs)>>{core::soup<decltype(segs)>{segs}};
}

template <typename Policy> auto make_view(const tf::segments<Policy> &obj) {
  return obj;
}

} // namespace tf
