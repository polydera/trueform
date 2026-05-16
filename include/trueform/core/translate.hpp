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
#include "./points.hpp"
#include "./policy/frame.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief In-place translation of a points view.
///
/// Mutates the underlying points buffer through the view by adding the
/// translation vector to each point.
///
/// @pre The view must not carry a frame policy. Tagged frames are
///      composed at read time, so mutating the underlying points would
///      silently apply the translation twice.
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::points<Policy> &view,
               const tf::vector_like<Dims, T> &v) {
  static_assert(!tf::has_frame_policy<Policy>,
                "tf::translate: view must not carry a frame policy.");
  tf::parallel_for_each(
      view, [&v](auto &&p) { p += v; }, tf::checked);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::points<Policy> &&view,
               const tf::vector_like<Dims, T> &v) {
  tf::translate(view, v);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::polygons<Policy> &view,
               const tf::vector_like<Dims, T> &v) {
  tf::translate(view.points(), v);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::polygons<Policy> &&view,
               const tf::vector_like<Dims, T> &v) {
  tf::translate(view.points(), v);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::segments<Policy> &view,
               const tf::vector_like<Dims, T> &v) {
  tf::translate(view.points(), v);
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
void translate(tf::segments<Policy> &&view,
               const tf::vector_like<Dims, T> &v) {
  tf::translate(view.points(), v);
}

} // namespace tf
