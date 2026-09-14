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
#include "../../core/checked.hpp"
#include "../../core/policy/frame.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/transformed.hpp"
#include "../volume.hpp"
#include <utility>

namespace tf {
namespace volume_detail {

/// @brief Move emitted geometry into the frame the volume states; a volume
/// with no frame emits local, untouched.
template <typename Points, typename Policy>
inline auto emit_in_stated_frame(Points points, const tf::volume<Policy> &vol)
    -> void {
  if constexpr (tf::has_frame_policy<Policy>) {
    const auto &pose = vol.frame();
    tf::parallel_for_each(
        points, [&pose](auto p) { p = tf::transformed(p, pose); }, tf::checked);
  } else {
    (void)points;
    (void)vol;
  }
}

/// @brief The mesh emission: points move through the stated frame, and a
/// reflecting frame (negative determinant) swaps each triangle's corners so
/// the extractor's outward-winding promise survives the mirror.
template <typename Index, typename Real, typename Policy>
inline auto emit_mesh_in_stated_frame(
    tf::polygons_buffer<Index, Real, 3, 3> &mesh,
    const tf::volume<Policy> &vol) -> void {
  if constexpr (tf::has_frame_policy<Policy>) {
    emit_in_stated_frame(mesh.points(), vol);
    const auto &t = vol.frame().transformation();
    const auto det = t(0, 0) * (t(1, 1) * t(2, 2) - t(1, 2) * t(2, 1)) -
                     t(0, 1) * (t(1, 0) * t(2, 2) - t(1, 2) * t(2, 0)) +
                     t(0, 2) * (t(1, 0) * t(2, 1) - t(1, 1) * t(2, 0));
    if (det < decltype(det){0})
      tf::parallel_for_each(
          mesh.faces_buffer(),
          [](auto face) { std::swap(face[0], face[1]); }, tf::checked);
  } else {
    (void)mesh;
    (void)vol;
  }
}

} // namespace volume_detail
} // namespace tf
