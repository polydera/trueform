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

#include "./aabb.hpp"
#include <limits>

namespace tf {

/// @ingroup core_primitives
/// @brief Create an empty (invalid) axis-aligned bounding box.
///
/// Returns an AABB with min > max, suitable for incremental
/// bounding box computation via union operations.
///
/// @tparam T The scalar type.
/// @tparam Dims The coordinate dimensions.
/// @return An empty @ref tf::aabb.
template <typename T, std::size_t Dims> auto make_empty_aabb() {
  tf::aabb<T, Dims> out;
  for (std::size_t i = 0; i < Dims; ++i) {
    out.min[i] = std::numeric_limits<T>::max();
    out.max[i] = std::numeric_limits<T>::lowest();
  }
  return out;
}
} // namespace tf
