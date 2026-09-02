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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/faces.hpp"
#include "../../core/views/zip.hpp"
#include <algorithm>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Reverse every selected face, in one pass over the faces.
/// @tparam Policy The faces policy type.
/// @tparam Range The flip mask range type.
/// @param faces The faces range (modified in place).
/// @param flip_mask Per-face reversal, aligned one-to-one with the faces.
template <typename Policy, typename Range>
auto apply_face_flip_mask(tf::faces<Policy> &faces, const Range &flip_mask)
    -> void {
  tf::parallel_for_each(tf::zip(faces, flip_mask), [](auto pair) {
    auto &&[face, flip] = pair;
    if (flip)
      std::reverse(face.begin(), face.end());
  });
}
} // namespace tf::topology
