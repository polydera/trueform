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
#include "./algorithm/parallel_for_each.hpp"
#include "./checked.hpp"
#include "./frame_like.hpp"
#include "./points.hpp"
#include "./policy/frame.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./transformation_like.hpp"
#include "./transformed.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief In-place affine transform of a points view.
///
/// Mutates the underlying points buffer through the view.
///
/// @pre The view must not carry a frame policy. Tagged frames are
///      composed at read time, so mutating the underlying points would
///      silently apply the transform twice.
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::points<Policy> &view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::parallel_for_each(
      view, [&t](auto &&p) { p = tf::transformed(p, t); }, tf::checked);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::points<Policy> &&view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::transform(view, t);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::polygons<Policy> &view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::transform(view.points(), t);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::polygons<Policy> &&view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::transform(view.points(), t);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::segments<Policy> &view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::transform(view.points(), t);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::segments<Policy> &&view,
               const tf::transformation_like<Dims, U> &t) -> void {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::transform: view must not carry a frame policy.");
  tf::transform(view.points(), t);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::points<Policy> &view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::points<Policy> &&view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::polygons<Policy> &view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::polygons<Policy> &&view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::segments<Policy> &view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transform(tf::segments<Policy> &&view,
               const tf::frame_like<Dims, U> &frame) -> void {
  tf::transform(view, frame.transformation());
}

} // namespace tf
