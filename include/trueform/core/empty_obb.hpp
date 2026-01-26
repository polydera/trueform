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

#include "./obb.hpp"
#include "./unit_vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Create an empty oriented bounding box.
///
/// Returns an OBB with zero extents and axis-aligned orientation.
///
/// @tparam T The scalar type.
/// @tparam Dims The coordinate dimensions.
/// @return An empty @ref tf::obb.
template <typename T, std::size_t Dims> auto make_empty_obb() {
  tf::obb<T, Dims> out{};
  for (std::size_t i = 0; i < Dims; ++i) {
    std::array<T, Dims> axis{};
    axis[i] = T(1);
    out.axes[i] = tf::make_unit_vector(tf::unsafe, tf::make_vector(axis));
    out.extent[i] = T(0);
  }
  return out;
}

} // namespace tf
