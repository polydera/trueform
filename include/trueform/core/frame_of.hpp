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
#include "./policy/frame.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Get the coordinate frame of a point set.
///
/// Returns the tagged frame if present, otherwise returns identity frame.
///
/// @tparam Policy The points policy type.
/// @param t The point set to query.
/// @return The frame or @ref tf::identity_frame if none tagged.
template <typename Policy>
auto frame_of(const tf::points<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a polygon mesh.
/// @overload
template <typename Policy>
auto frame_of(const tf::polygons<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a vector set.
/// @overload
template <typename Policy>
auto frame_of(const tf::vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a unit vector set.
/// @overload
template <typename Policy>
auto frame_of(const tf::unit_vectors<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<typename Policy::value_type>>{};
  }
}

/// @ingroup core_properties
/// @brief Get the coordinate frame of a segment set.
/// @overload
template <typename Policy>
auto frame_of(const tf::segments<Policy> &t) -> decltype(auto) {
  if constexpr (has_frame_policy<Policy>)
    return t.frame();
  else {
    return tf::identity_frame<tf::coordinate_type<Policy>,
                              tf::static_size_v<decltype(t.points()[0])>>{};
  }
}

} // namespace tf
