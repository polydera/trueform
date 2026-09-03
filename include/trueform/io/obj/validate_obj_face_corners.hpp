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

#include "../../core/algorithm/reduce.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/mapped_range.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief States whether every parsed corner names a point in
/// `[0, n_points)`.
template <typename Corners>
auto validate_obj_face_corners(const Corners &corners, std::size_t n_points)
    -> bool {
  auto known = tf::make_mapped_range(corners, [n_points](auto corner) {
    return corner >= 0 && static_cast<std::size_t>(corner) < n_points;
  });
  const auto both = [](bool left, bool right) { return left && right; };
  return tf::reduce(known, both, true, tf::checked);
}

} // namespace tf::io::obj
