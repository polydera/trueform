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
#include "./base/curves.hpp"
#include "./paths.hpp"
#include "./points.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief A collection of curves sharing point data.
///
/// Wraps paths (connectivity) and points to provide curve iteration.
/// Each element is a view over the underlying data.
///
/// @tparam Policy The underlying curves policy.
template <typename Policy> struct curves : Policy {
  curves(const Policy &r) : Policy{r} {}
  curves(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const curves<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(curves<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(curves<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const curves<Policy> &, T &&t) {
  return curves<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(curves<Policy> &, T &&t) {
  return curves<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(curves<Policy> &&, T &&t) {
  return curves<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Create curves from paths and points.
///
/// @tparam Range0 The paths range type (e.g., offset block range).
/// @tparam Range1 The points range type.
/// @param paths The path connectivity data.
/// @param points The point data.
/// @return A @ref tf::curves.
template <typename Range0, typename Range1>
auto make_curves(Range0 &&paths, Range1 &&points) {
  auto r0 = tf::make_paths(paths);
  auto r1 = tf::make_points(points);
  return curves<core::curves<decltype(r0), decltype(r1)>>{
      core::curves<decltype(r0), decltype(r1)>{r0, r1}};
}

template <typename Range> auto make_curves(curves<Range> p) -> curves<Range> {
  return p;
}

/// @ingroup core_ranges
/// @brief Create curves from a range of curve objects.
/// @overload
template <typename Range> auto make_curves(Range &&r) {
  auto segs = tf::make_range(r);
  return curves<decltype(segs)>{segs};
}

template <typename Policy> auto make_view(const tf::curves<Policy> &obj) {
  return obj;
}

} // namespace tf
