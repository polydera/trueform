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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/intersects.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/segments.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/segment_intersect.hpp"
#include "../../exact/vertex.hpp"
#include "../exact/predicate_kernel.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/search_self.hpp"
#include "../intersect_mode.hpp"
#include "./crossing_record.hpp"
#include "./edge.hpp"

#include <algorithm>
#include <array>

namespace tf::intersect::graph {

/// Test one pair of edges for intersection. `ea`, `eb` are the edge
/// instances; `eid_a`, `eid_b` are canonical group IDs (`edge.id`) used
/// in the records.
template <typename Index, typename GetPoint, typename Int>
auto test_edge_pair(const edge<Index> &ea, const edge<Index> &eb, Index eid_a,
                    Index eid_b, int ax0, int ax1, const GetPoint &get_point,
                    tf::buffer<crossing_record<Index>> &out,
                    const tf::exact::predicate_kernel<Int> &kernel) -> void {
  auto pa0 = get_point(-1, ea.point_0);
  auto pa1 = get_point(-1, ea.point_1);
  if (pa0[0] == pa1[0] && pa0[1] == pa1[1] && pa0[2] == pa1[2])
    return;
  auto pb0 = get_point(-1, eb.point_0);
  auto pb1 = get_point(-1, eb.point_1);
  if (pb0[0] == pb1[0] && pb0[1] == pb1[1] && pb0[2] == pb1[2])
    return;

  auto lo = std::min(eid_a, eid_b), hi = std::max(eid_a, eid_b);

  tf::exact::vertex<Index, Int> va0{ea.point_0, pa0}, va1{ea.point_1, pa1};
  tf::exact::vertex<Index, Int> vb0{eb.point_0, pb0}, vb1{eb.point_1, pb1};

  auto hits =
      tf::exact::classify_segments(va0, va1, vb0, vb1, ax0, ax1, kernel);
  if (!hits)
    return;

  auto emit = [&](const tf::exact::segment_intersection<short> &h) {
    if (h.target_a.label == tf::topo_type::edge &&
        h.target_b.label == tf::topo_type::edge) {
      std::array<face_id<Index>, 4> faces = {
          face_id<Index>{ea.tag, ea.object},
          face_id<Index>{ea.tag_other, ea.object_other},
          face_id<Index>{eb.tag, eb.object},
          face_id<Index>{eb.tag_other, eb.object_other}};
      std::sort(faces.begin(), faces.end());
      (void)std::unique(faces.begin(), faces.end());
      out.push_back({lo,
                     hi,
                     {},
                     {},
                     {faces[0], faces[1], faces[2]},
                     crossing_record<Index>::ee});
    } else if (h.target_a.label == tf::topo_type::vertex &&
               h.target_b.label == tf::topo_type::vertex) {
      Index pa = h.target_a.id == 0 ? ea.point_0 : ea.point_1;
      Index pb = h.target_b.id == 0 ? eb.point_0 : eb.point_1;
      out.push_back({lo,
                     hi,
                     std::min(pa, pb),
                     std::max(pa, pb),
                     {},
                     crossing_record<Index>::vv});
    } else {
      Index edge_id, point_id;
      if (h.target_a.label == tf::topo_type::vertex) {
        edge_id = eid_b;
        point_id = h.target_a.id == 0 ? ea.point_0 : ea.point_1;
      } else {
        edge_id = eid_a;
        point_id = h.target_b.id == 0 ? eb.point_0 : eb.point_1;
      }
      out.push_back(
          {edge_id, {}, point_id, {}, {}, crossing_record<Index>::ve});
    }
  };

  emit(hits->first);
  if (hits->second)
    emit(*hits->second);
}

template <typename Index>
auto has_a_single_contour(const Index *edge_ids, std::size_t k,
                          const tf::buffer<edge<Index>> &edge_data) {
  if (k <= 1) return true;
  auto t = edge_data[edge_ids[0]].tag_other;
  for (std::size_t i = 1; i < k; ++i)
    if (edge_data[edge_ids[i]].tag_other != t)
      return false;
  return true;
}

/// Detect crossings on one face — brute force (k <= 8).
///
/// edge_ids are instance indices into edge_data (the flat buffer).
/// Canonical group IDs come from edge.id.
template <typename Index, typename GetPoint, typename Int>
auto detect_crossings_brute(const Index *edge_ids, std::size_t k,
                            const tf::buffer<edge<Index>> &edge_data, int ax0,
                            int ax1, const GetPoint &get_point,
                            tf::buffer<crossing_record<Index>> &out,
                            tf::intersect_mode mode,
                            const tf::exact::predicate_kernel<Int> &kernel)
    -> void {
  if (!(mode & tf::intersect_mode::resolve_self_crossing_contours) &&
      has_a_single_contour(edge_ids, k, edge_data))
    return;
  for (std::size_t i = 0; i < k; ++i) {
    auto &&ea = edge_data[edge_ids[i]];
    for (std::size_t j = i + 1; j < k; ++j) {
      auto &&eb = edge_data[edge_ids[j]];
      if (ea.tag_other == eb.tag_other &&
          !(mode & tf::intersect_mode::resolve_self_crossing_contours))
        continue;
      test_edge_pair<Index>(ea, eb, ea.id, eb.id, ax0, ax1, get_point, out,
                            kernel);
    }
  }
}

/// Detect crossings on one face — AABB tree (k > 8).
///
/// edge_ids are instance indices into edge_data (the flat buffer).
template <typename Index, typename GetPoint, typename Int>
auto detect_crossings_tree(const Index *edge_ids, std::size_t k,
                           const tf::buffer<edge<Index>> &edge_data, int ax0,
                           int ax1, const GetPoint &get_point,
                           tf::buffer<crossing_record<Index>> &out,
                           tf::points_buffer<Int, 2> &pts_2d,
                           tf::buffer<int> &edge_pairs,
                           tf::aabb_tree<int, Int, 2> &tree,
                           tf::intersect_mode mode,
                           const tf::exact::predicate_kernel<Int> &kernel)
    -> void {
  if (!(mode & tf::intersect_mode::resolve_self_crossing_contours) &&
      has_a_single_contour(edge_ids, k, edge_data))
    return;
  pts_2d.clear();
  edge_pairs.clear();
  pts_2d.reserve(k * 2);
  edge_pairs.reserve(k * 2);
  for (std::size_t i = 0; i < k; ++i) {
    auto &&e = edge_data[edge_ids[i]];
    auto p0 = get_point(-1, e.point_0);
    auto p1 = get_point(-1, e.point_1);
    pts_2d.emplace_back(p0[ax0], p0[ax1]);
    pts_2d.emplace_back(p1[ax0], p1[ax1]);
    edge_pairs.push_back(static_cast<int>(2 * i));
    edge_pairs.push_back(static_cast<int>(2 * i + 1));
  }
  auto segs = tf::make_segments(edge_pairs, pts_2d);
  tree.build(segs, tf::config_tree(4, 4));
  tf::search_self(
      tree, tf::intersects_f,
      [&](int id0, int id1) {
        auto &&ea = edge_data[edge_ids[id0]];
        auto &&eb = edge_data[edge_ids[id1]];
        if (ea.tag_other == eb.tag_other &&
            !(mode & tf::intersect_mode::resolve_self_crossing_contours))
          return;
        test_edge_pair<Index>(ea, eb, ea.id, eb.id, ax0, ax1, get_point, out,
                              kernel);
      },
      /*parallelism_depth=*/0);
}

/// Gather crossing records across all faces (parallel per face).
///
/// _edges stores instance indices into _edge_defs.data_buffer().
/// Each instance carries its own (tag, tag_other) for correct triple
/// construction even with duplicate triangles.
template <typename Index, typename ApplyToFace, typename GetPoint, typename Int>
auto gather_crossing_records(
    const tf::offset_block_buffer<Index, Index> &edges,
    const tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
    const ApplyToFace &apply_to_face, const GetPoint &get_point,
    tf::intersect_mode mode,
    const tf::exact::predicate_kernel<Int> &kernel)
    -> tf::buffer<crossing_record<Index>> {
  struct local_t {
    tf::buffer<crossing_record<Index>> records;
    tf::points_buffer<Int, 2> pts_2d;
    tf::buffer<int> edge_pairs;
    tf::aabb_tree<int, Int, 2> tree;
  };

  tf::buffer<crossing_record<Index>> out;
  auto &edge_data = edge_defs.data_buffer();

  auto task = [&](auto &&range, local_t &local) {
    auto get_point_f = get_point;
    auto apply_to_face_f = apply_to_face;
    for (const auto &face_edges : range) {
      auto k = face_edges.size();
      if (k < 2)
        continue;
      // Instance index → edge instance → face polygon for projection axes
      auto &&inst = edge_data[face_edges[0]];
      apply_to_face_f(inst.tag, inst.object, [&](const auto &face) {
        auto fp0 = get_point_f(inst.tag, face[0]);
        auto fp1 = get_point_f(inst.tag, face[1]);
        auto fp2 = get_point_f(inst.tag, face[2]);
        auto [ax0, ax1] = tf::exact::projection_axes(fp0, fp1, fp2);

        if (k <= 8)
          detect_crossings_brute<Index>(face_edges.begin(), k, edge_data, ax0,
                                        ax1, get_point_f, local.records, mode,
                                        kernel);
        else
          detect_crossings_tree<Index>(
              face_edges.begin(), k, edge_data, ax0, ax1, get_point_f,
              local.records, local.pts_2d, local.edge_pairs, local.tree, mode,
              kernel);
      });
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.records, out);
  };

  tf::blocked_reduce_sequenced_aggregate(edges, tf::none, local_t{}, task, agg);
  return out;
}

} // namespace tf::intersect::graph
