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
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./polygons.hpp"
#include "./polygons_buffer.hpp"
#include "./segments.hpp"
#include "./segments_buffer.hpp"
#include "./translate.hpp"
#include "./vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Copy-returning translation of a polygons view.
///
/// Copies the view into an owning @ref tf::polygons_buffer, then
/// applies @ref tf::translate in place on the new buffer's points.
template <typename Policy, std::size_t Dims, typename T>
auto translated(const tf::polygons<Policy> &view,
                const tf::vector_like<Dims, T> &v) {
  auto out = tf::make_polygons_buffer(view);
  tf::translate(out.points(), v);
  return out;
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
auto translated(const tf::segments<Policy> &view,
                const tf::vector_like<Dims, T> &v) {
  auto out = tf::make_segments_buffer(view);
  tf::translate(out.points(), v);
  return out;
}

/// @ingroup core_primitives
/// @overload
template <typename Policy, std::size_t Dims, typename T>
auto translated(const tf::points<Policy> &view,
                const tf::vector_like<Dims, T> &v) {
  auto out = tf::make_points_buffer(view);
  tf::translate(out.points(), v);
  return out;
}

} // namespace tf
