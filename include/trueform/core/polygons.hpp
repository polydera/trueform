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
#include "./base/polygons.hpp"
#include "./base/soup.hpp"
#include "./faces.hpp"
#include "./form.hpp"
#include "./points.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief A range of polygons.
///
/// Wraps a range policy providing access to polygon primitives.
/// Inherits from form to enable policy composition with spatial trees,
/// frames, and other policies.
///
/// @tparam Policy The underlying range policy.
template <typename Policy>
struct polygons : form<coordinate_dims_v<Policy>, Policy> {
  using base = form<coordinate_dims_v<Policy>, Policy>;
  polygons(const Policy &r) : base{r} {}
  polygons(Policy &&r) : base{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const polygons<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(polygons<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(polygons<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const polygons<Policy> &, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(polygons<Policy> &, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(polygons<Policy> &&, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create a range of polygons from faces and points.
///
/// @tparam Range0 The face index range type.
/// @tparam Range1 The point range type.
/// @param faces A range of face index arrays.
/// @param points A range of points.
/// @return A @ref tf::polygons range.
template <typename Range0, typename Range1>
auto make_polygons(Range0 &&faces, Range1 &&points) {
  auto r0 = tf::make_faces(faces);
  auto r1 = tf::make_points(points);
  return polygons<core::polygons<decltype(r0), decltype(r1)>>{
      core::polygons<decltype(r0), decltype(r1)>{r0, r1}};
}

/// @ingroup core_ranges
/// @brief Identity overload for already-wrapped polygon ranges.
template <typename Range>
auto make_polygons(polygons<Range> p) -> polygons<Range> {
  return p;
}

/// @ingroup core_ranges
/// @brief Create a polygon soup from a generic range.
///
/// Wraps a range of polygon primitives into a @ref tf::polygons range
/// using the soup adapter.
template <typename Range> auto make_polygons(Range &&r) {
  auto polys = tf::make_range(r);
  return polygons<core::soup<decltype(polys)>>{
      core::soup<decltype(polys)>{std::move(polys)}};
}

template <typename Policy> auto make_view(const tf::polygons<Policy> &obj) {
  return obj;
}

} // namespace tf
