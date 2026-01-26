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

#include "./rss.hpp"
#include "./unit_vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Create an empty rectangle swept sphere.
///
/// Returns an RSS with zero lengths and radius.
///
/// @tparam T The scalar type.
/// @tparam Dims The coordinate dimensions.
/// @return An empty @ref tf::rss.
template <typename T, std::size_t Dims> auto make_empty_rss() {
  tf::rss<T, Dims> out{};
  for (std::size_t i = 0; i < Dims; ++i) {
    std::array<T, Dims> axis{};
    axis[i] = T(1);
    out.axes[i] = tf::make_unit_vector(tf::unsafe, tf::make_vector(axis));
  }
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    out.length[i] = T(0);
  }
  out.radius = T(0);
  return out;
}

} // namespace tf
