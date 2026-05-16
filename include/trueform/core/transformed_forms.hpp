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
#include "./frame_like.hpp"
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./polygons.hpp"
#include "./polygons_buffer.hpp"
#include "./segments.hpp"
#include "./segments_buffer.hpp"
#include "./transform.hpp"
#include "./transformation_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Copy-returning affine transform of a polygons view.
///
/// Copies the view into an owning @ref tf::polygons_buffer, then
/// applies @ref tf::transform in place on the new buffer's points.
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::polygons<Policy> &view,
                 const tf::transformation_like<Dims, U> &t) {
  auto out = tf::make_polygons_buffer(view);
  tf::transform(out.points(), t);
  return out;
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::polygons<Policy> &view,
                 const tf::frame_like<Dims, U> &frame) {
  return tf::transformed(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::segments<Policy> &view,
                 const tf::transformation_like<Dims, U> &t) {
  auto out = tf::make_segments_buffer(view);
  tf::transform(out.points(), t);
  return out;
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::segments<Policy> &view,
                 const tf::frame_like<Dims, U> &frame) {
  return tf::transformed(view, frame.transformation());
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::points<Policy> &view,
                 const tf::transformation_like<Dims, U> &t) {
  auto out = tf::make_points_buffer(view);
  tf::transform(out.points(), t);
  return out;
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename U>
auto transformed(const tf::points<Policy> &view,
                 const tf::frame_like<Dims, U> &frame) {
  return tf::transformed(view, frame.transformation());
}

} // namespace tf
