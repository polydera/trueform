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

#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"
#include "./face_plane_info.hpp"
#include "./predicate_kernel.hpp"

namespace tf::exact {

/// Vertex-face: representative source vertices with sign==0 lying
/// strictly inside the target face polygon (in 2D projection).
/// A vertex within the kernel's band of any target edge yields sign 0
/// and is rejected here so it falls through to the VE branch in
/// `coplanar_primitives`.
template <typename Index, typename Int, typename SourceVRep,
          typename SourceShared, typename Intersections, typename Pts,
          typename Kernel = tf::exact::predicate_kernel<Int>>
void vertex_face(const tf::buffer<tf::exact::vertex<Index, Int>> &source_verts,
                 std::size_t n_source,
                 const tf::buffer<tf::exact::vertex<Index, Int>> &target_verts,
                 std::size_t n_target,
                 const tf::small_vector<int, 16> &source_signs,
                 int source_tag, int target_tag, Index source_face_id,
                 Index target_face_id, const SourceVRep &source_vrep,
                 const SourceShared &source_shared,
                 const face_plane_info &target_plane,
                 Intersections &intersections, Pts &pts,
                 const Kernel &kernel = {}) {
  if (!target_plane.valid)
    return;
  int ax0 = target_plane.ax0, ax1 = target_plane.ax1;
  for (std::size_t i = 0; i < n_source; ++i) {
    if (source_shared(i))
      continue;
    if (source_signs[i] != 0)
      continue;
    if (!source_vrep(i))
      continue;
    int first_sign = 0;
    bool inside = true;
    for (std::size_t k = 0; k < n_target && inside; ++k) {
      auto nk = tf::circular_increment(k, n_target);
      auto s = kernel.orient2d_sign(target_verts[k], target_verts[nk],
                                    source_verts[i], ax0, ax1);
      if (s == 0)
        inside = false;
      else if (first_sign == 0)
        first_sign = s;
      else if (s != first_sign)
        inside = false;
    }
    if (inside && first_sign != 0)
      emit_record(source_tag, target_tag, source_face_id, target_face_id,
                  {Index(i), tf::topo_type::vertex},
                  {target_face_id, tf::topo_type::face}, source_verts[i].pt,
                  intersections, pts);
  }
}

template <typename Index, typename Int, typename SourceVRep,
          typename Intersections, typename Pts,
          typename Kernel = tf::exact::predicate_kernel<Int>>
void vertex_face(const tf::buffer<tf::exact::vertex<Index, Int>> &source_verts,
                 std::size_t n_source,
                 const tf::buffer<tf::exact::vertex<Index, Int>> &target_verts,
                 std::size_t n_target,
                 const tf::small_vector<int, 16> &source_signs,
                 int source_tag, int target_tag, Index source_face_id,
                 Index target_face_id, const SourceVRep &source_vrep,
                 const face_plane_info &target_plane,
                 Intersections &intersections, Pts &pts,
                 const Kernel &kernel = {}) {
  vertex_face(source_verts, n_source, target_verts, n_target, source_signs,
              source_tag, target_tag, source_face_id, target_face_id,
              source_vrep, [](std::size_t) { return false; }, target_plane,
              intersections, pts, kernel);
}

} // namespace tf::exact
