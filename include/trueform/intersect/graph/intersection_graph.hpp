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
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "./canonicalize_edges.hpp"
#include "./crossing_classification.hpp"
#include "./crossing_detection.hpp"
#include "./crossing_points.hpp"
#include "./crossing_split_entries.hpp"
#include "./crossing_vv_remap.hpp"
#include "./edge.hpp"
#include "./edge_extractor.hpp"
#include "./loop.hpp"
#include "./split_edges.hpp"
#include "./vertex.hpp"

namespace tf {

/// Builds split boundary loops and interior contour edges for intersected
/// faces.
///
/// For each intersection subrange (grouped by (tag, object)):
/// - produces a base loop: the original face boundary with intersection
///   points inserted in parametric order along each edge.
/// - produces interior contour edges: one per (tag_other, object_other)
///   face pair that has exactly 2 records (the two endpoints).
///
/// Boundary edges where both endpoints are consecutive face vertices are
/// expanded using the base loop, splitting into sub-edges through any
/// intermediate intersection points.
///
/// After building, edges are canonicalized and crossings are detected
/// and resolved (EE splits, VE splits, VV merges).
template <typename Index> class intersection_graph {
public:
  auto loops() const { return tf::make_range(_loops); }
  auto edges() const { return tf::make_range(_edges); }
  auto edge_groups() const { return tf::make_range(_edge_defs); }
  auto crossing_points() const { return tf::make_range(_crossing_points); }
  auto crossing_base() const { return _crossing_base; }
  auto point_remap() const { return tf::make_range(_point_remap); }

  auto clear() {
    _loops.clear();
    _edges.clear();
    _edge_defs.clear();
    _crossing_points.clear();
    _point_remap.clear();
    _crossing_base = {};
  }

  /// Build loops, edges, canonicalize, and resolve crossings.
  template <typename Subranges, typename GetFace, typename GetPoint,
            typename GetFlatId>
  auto build(const Subranges &subranges, const GetFace &get_face,
             const GetPoint &get_point, const GetFlatId &get_flat_id) -> void {
    clear();
    if (subranges.size() == 0)
      return;
    build_loops(subranges, get_face, get_point, get_flat_id);
    build_edges(subranges, get_face);
    intersect::graph::canonicalize_edges(_edge_defs, _edges);
    detect_and_split_crossings(get_face, get_point);
  }

private:
  template <typename Subranges, typename GetFace, typename GetPoint,
            typename GetFlatId>
  auto build_loops(const Subranges &subranges, const GetFace &get_face,
                   const GetPoint &get_point, const GetFlatId &get_flat_id)
      -> void {
    auto n = subranges.size();
    _loops.offsets_buffer().allocate(n + 1);
    _loops.offsets_buffer()[0] = 0;

    std::size_t loop_i = 1;

    struct local_t {
      tf::buffer<Index> sizes;
      tf::buffer<intersect::graph::vertex<Index>> data;
      tf::buffer<intersect::graph::loop_node<Index>> work;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto get_face_f = get_face;
      auto get_point_f = get_point;
      auto get_flat_id_f = get_flat_id;
      local.sizes.allocate(range.size());
      auto sit = local.sizes.begin();
      for (const auto &subrange : range) {
        auto old_size = local.data.size();
        if (subrange.size() != 0) {
          auto tag = subrange[0].tag;
          auto object = subrange[0].object;
          intersect::graph::extract_loop<Index>(subrange, get_face_f(tag, object), tag,
                              get_point_f, get_flat_id_f, local.work,
                              local.data);
        }
        *sit++ = static_cast<Index>(local.data.size() - old_size);
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.data, _loops.data_buffer());
      for (auto sz : local.sizes) {
        _loops.offsets_buffer()[loop_i] =
            _loops.offsets_buffer()[loop_i - 1] + sz;
        ++loop_i;
      }
    };

    tf::blocked_reduce_sequenced_aggregate(subranges, tf::none, local_t{},
                                           task, agg);
  }

  template <typename Subranges, typename GetFace>
  auto build_edges(const Subranges &subranges, const GetFace &get_face)
      -> void {
    auto n = subranges.size();
    _edges.offsets_buffer().allocate(n + 1);
    _edges.offsets_buffer()[0] = 0;

    std::size_t edge_i = 1;
    auto all_loops = tf::make_range(_loops);

    struct local_t {
      tf::buffer<Index> counts;
      tf::buffer<intersect::graph::edge<Index>> data;
      intersect::graph::edge_extractor<Index> extractor;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto get_face_f = get_face;
      local.counts.allocate(range.size());
      auto cit = local.counts.begin();
      for (const auto &subrange : range) {
        auto old_size = local.data.size();
        if (subrange.size() != 0) {
          auto tag = subrange[0].tag;
          auto object = subrange[0].object;
          auto face_size = get_face_f(tag, object).size();
          local.extractor.extract(subrange, face_size, all_loops,
                                   subranges, get_face_f, local.data);
        }
        *cit++ = static_cast<Index>(local.data.size() - old_size);
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      auto &edge_defs_raw = _edge_defs.data_buffer();
      auto old_ed_size = edge_defs_raw.size();
      edge_defs_raw.reallocate(old_ed_size + local.data.size());
      auto &edges_raw = _edges.data_buffer();
      auto old_e_size = edges_raw.size();
      edges_raw.reallocate(old_e_size + local.data.size());
      auto offset = static_cast<Index>(old_ed_size);
      auto it0 = edge_defs_raw.begin() + old_ed_size;
      auto it1 = edges_raw.begin() + old_e_size;
      for (auto e : local.data) {
        e.id += offset;
        *it0++ = e;
        *it1++ = e.id;
      }
      for (auto sz : local.counts) {
        _edges.offsets_buffer()[edge_i] =
            _edges.offsets_buffer()[edge_i - 1] + sz;
        ++edge_i;
      }
    };

    tf::blocked_reduce_sequenced_aggregate(subranges, tf::none, local_t{},
                                           task, agg);
  }

  template <typename GetFace, typename GetPoint>
  auto detect_and_split_crossings(const GetFace &get_face,
                                  const GetPoint &get_point) -> void {
    if (_edges.size() == 0)
      return;

    Index max_point_id = 0;
    for (auto &&grp : _edge_defs)
      for (auto &&e : grp)
        max_point_id = std::max(max_point_id, std::max(e.point_0, e.point_1));

    // 1. Gather crossing records (parallel per face)
    auto records =
        intersect::graph::gather_crossing_records<Index>(
            _edges, _edge_defs, get_face, get_point);
    if (records.size() == 0)
      return;

    // 2. Sort by (type, type-specific key)
    intersect::graph::sort_crossing_records<Index>(records);

    // 3. Find type boundaries
    auto [ee_range, ve_range, vv_range] =
        intersect::graph::find_type_ranges<Index>(records);
    // 4. EE: group by triple, compute crossing point coordinates
    auto ee_offsets = intersect::graph::compute_ee_crossing_points<Index>(
        ee_range, max_point_id, _edge_defs, get_point, _crossing_base,
        _crossing_points);
    auto num_unique_ee =
        ee_offsets.size() > 0 ? ee_offsets.size() - 1 : std::size_t(0);

    // 5. VE: deduplicate
    ve_range = intersect::graph::dedup_ve(ve_range);

    // 6. VV: deduplicate + union-find remap
    bool has_vv = intersect::graph::apply_vv_remap(
        vv_range, max_point_id + 1 + static_cast<Index>(num_unique_ee),
        _edge_defs, _loops, _point_remap);
    // 7. Collect split entries
    auto entries = intersect::graph::collect_split_entries<Index>(
        ee_range, ee_offsets, ve_range, _crossing_base);
    if (entries.size() == 0) {
      if (has_vv)
        tf::intersect::graph::canonicalize_edges(_edge_defs, _edges);
      return;
    }

    // 8. Split edges and rebuild
    intersect::graph::split_edges<Index>(
        entries, has_vv, _point_remap, _crossing_base, _crossing_points,
        _edge_defs, _edges, get_point);
  }

  tf::offset_block_buffer<Index, intersect::graph::vertex<Index>> _loops;
  tf::offset_block_buffer<Index, Index> _edges;
  tf::offset_block_buffer<Index, intersect::graph::edge<Index>> _edge_defs;
  tf::buffer<tf::point<int32_t, 3>> _crossing_points;
  tf::buffer<Index> _point_remap;
  Index _crossing_base = {};
};

} // namespace tf
