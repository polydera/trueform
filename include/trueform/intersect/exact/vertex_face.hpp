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

namespace tf::exact {

/// Vertex-face: check if representative source vertices with sign==0 lie
/// strictly inside the target face polygon (in 2D projection).
template <typename Index, typename SourceVRep, typename Ints, typename Pts>
void vertex_face(const tf::buffer<tf::exact::vertex<Index>> &source_verts, std::size_t n_source,
                 const tf::buffer<tf::exact::vertex<Index>> &target_verts, std::size_t n_target,
                 const tf::small_vector<int, 16> &source_signs, int source_tag,
                 int target_tag, Index source_face_id, Index target_face_id,
                 const SourceVRep &source_vrep,
                 const face_plane_info &target_plane, Ints &ints, Pts &pts) {
  if (!target_plane.valid)
    return;
  int ax0 = target_plane.ax0, ax1 = target_plane.ax1;
  for (std::size_t i = 0; i < n_source; ++i) {
    if (source_signs[i] != 0)
      continue;
    if (!source_vrep(i))
      continue;
    int first_sign = 0;
    bool inside = true;
    for (std::size_t k = 0; k < n_target && inside; ++k) {
      auto nk = tf::circular_increment(k, n_target);
      auto s = orient2d_sign(target_verts[k], target_verts[nk],
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
                  ints, pts);
  }
}

} // namespace tf::exact
