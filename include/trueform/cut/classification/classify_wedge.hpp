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
#include "../../core/sidedness.hpp"
#include "../../exact/classify_wedge.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/edge_id_in_face.hpp"

namespace tf::cut {

/// Classify other-mesh faces at an intersection edge using exact wedge test.
///
/// For an intersection edge in loop0, finds each wedge partner (same-mesh
/// face sharing the edge in opposite direction), then classifies each
/// other-mesh face neighbor by testing a point against the wedge.
///
/// Coplanar edges (boundary result) are skipped.
///
/// Accumulates per-component vote counts:
///   0 = inside (on_negative_side)
///   1 = outside (on_positive_side)
template <typename Loop, typename Desc, typename Neighbors, typename AllLoops,
          typename AllDescs, typename GetPoint, typename ApplyToFace,
          typename IsCoplanar, typename Labels, typename Counts>
void classify_edge_by_wedge(const Loop &loop0, const Desc &desc0,
                            std::size_t edge_id, const Neighbors &neighbors,
                            const AllLoops &all_loops,
                            const AllDescs &all_descs,
                            const GetPoint &get_point,
                            const ApplyToFace &apply_to_face,
                            const IsCoplanar &is_coplanar, const Labels &labels,
                            Counts &local_counts) {

  auto n = loop0.size();
  auto next_edge_id = tf::circular_increment(edge_id, n);
  auto v0_id = loop0[edge_id];
  auto v1_id = loop0[next_edge_id];

  auto v0 = get_point(desc0.tag, v0_id);
  auto v1 = get_point(desc0.tag, v1_id);

  using Int = tf::coordinate_type<decltype(v0)>;
  using pt3 = tf::point<Int, 3>;
  using pt2 = tf::point<Int, 2>;
  using vertex_t = std::decay_t<decltype(loop0[0])>;

  // Find a face vertex to the left of directed edge (e0→e1) in projected 2D.
  auto find_left_vertex = [&](int tag, int object, const pt3 &e0,
                              const pt3 &e1) -> pt3 {
    pt3 result = e0; // degenerate fallback
    apply_to_face(tag, object, [&](const auto &face) {
      auto mk = [](auto id) {
        return vertex_t{tf::intersect::graph::vertex_source::original, id, {}};
      };
      auto p0 = get_point(tag, mk(face[0]));
      auto p1 = get_point(tag, mk(face[1]));
      auto p2 = get_point(tag, mk(face[2]));
      auto axes = tf::exact::projection_axes(p0, p1, p2);
      int ax0 = axes.first, ax1 = axes.second;
      pt2 e0_2d = {e0[ax0], e0[ax1]};
      pt2 e1_2d = {e1[ax0], e1[ax1]};
      for (std::size_t k = 0; k < face.size(); ++k) {
        auto fv = get_point(tag, mk(face[k]));
        pt2 fv_2d = {fv[ax0], fv[ax1]};
        if (tf::exact::orient2d(e0_2d, e1_2d, fv_2d) > 0) {
          result = fv;
          return;
        }
      }
    });
    return result;
  };

  // Face A: current face, edge direction v0→v1
  auto wedge_a = find_left_vertex(desc0.tag, desc0.object, v0, v1);

  // Classify all other-mesh neighbors against a wedge defined by (a, b)
  auto classify_with_wedge = [&](const pt3 &wa, const pt3 &wb) {
    for (auto n_id : neighbors) {
      auto &d_other = all_descs[n_id];
      if (d_other.tag == desc0.tag)
        continue;
      if (is_coplanar(n_id))
        continue;

      auto &&loop_other = all_loops[n_id];
      auto eid = tf::edge_id_in_face(v0_id, v1_id, loop_other);
      if (eid == loop_other.size())
        continue;

      // Test point: face vertex to the left of the edge from the
      // other face's perspective
      bool other_same_dir = (loop_other[eid] == v0_id);
      auto test_pt =
          other_same_dir
              ? find_left_vertex(d_other.tag, d_other.object, v0, v1)
              : find_left_vertex(d_other.tag, d_other.object, v1, v0);

      auto side = tf::exact::classify_wedge(v0, v1, wa, wb, test_pt);
      if (side == tf::sidedness::on_boundary)
        continue;

      int classification = (side == tf::sidedness::on_negative_side) ? 0 : 1;
      auto label = labels[n_id];
      if (label >= 0)
        local_counts[label][classification]++;
    }
  };

  // Find wedge partners: same tag, opposite edge direction
  for (auto wp_id : neighbors) {
    auto &d_wp = all_descs[wp_id];
    if (d_wp.tag != desc0.tag)
      continue;

    auto &&loop_wp = all_loops[wp_id];
    auto eid = tf::edge_id_in_face(v0_id, v1_id, loop_wp);
    if (eid == loop_wp.size())
      continue;
    if (loop_wp[eid] == v0_id)
      continue; // same direction — need opposite

    auto wedge_b = find_left_vertex(d_wp.tag, d_wp.object, v1, v0);
    classify_with_wedge(wedge_a, wedge_b);
  }
}

} // namespace tf::cut
