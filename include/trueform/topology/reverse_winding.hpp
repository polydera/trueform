/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/faces.hpp"
#include <algorithm>

namespace tf {

/// @ingroup topology_analysis
/// @brief Reverse the winding order of all faces.
///
/// Reverses the vertex order of each face, effectively flipping the
/// face normals. This changes the orientation of the mesh surface.
///
/// @tparam Policy The faces policy type.
/// @param faces The faces range (modified in place).
template <typename Policy>
auto reverse_winding(tf::faces<Policy> &faces) -> void {
  tf::parallel_apply(faces, [](auto &&face) {
    std::reverse(face.begin(), face.end());
  });
}

/// @ingroup topology_analysis
/// @overload
template <typename Policy>
auto reverse_winding(tf::faces<Policy> &&faces) -> void {
  reverse_winding(faces);
}
} // namespace tf
