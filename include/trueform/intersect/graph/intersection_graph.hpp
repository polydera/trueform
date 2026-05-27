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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/take.hpp"
#include "../exact/predicate_kernel.hpp"
#include "../exact/tagged_intersections.hpp"
#include "../intersect_mode.hpp"
#include "./canonicalize_edges.hpp"
#include "./clean_loop.hpp"
#include "./collect_vv_pairs.hpp"
#include "./crossing_classification.hpp"
#include "./crossing_detection.hpp"
#include "./crossing_points.hpp"
#include "./crossing_split_entries.hpp"
#include "./edge.hpp"
#include "./edge_extractor.hpp"
#include "./face_descriptor.hpp"
#include "./loop.hpp"
#include "./split_edges.hpp"
#include "./strip_base_loop_edges.hpp"
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
template <typename Index, typename Int = tf::exact::int32>
class intersection_graph {
public:
  auto points() const { return _points.points(); }
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto loops() const { return tf::make_range(_loops); }
  auto edge_ids() const { return tf::make_range(_edges); }
  auto edges() const {
    return tf::make_block_indirect_range(_edges, _edge_defs.data_buffer());
  }
  auto edge_groups() const { return tf::make_range(_edge_defs); }
  auto tag_offsets() const { return tf::make_range(_tag_offsets); }
  auto point_remap() const { return tf::make_range(_point_remap); }
  auto crossing_point_ids() const {
    return tf::make_range(_crossing_point_ids);
  }

  auto clear() {
    _points.clear();
    _crossing_point_ids.clear();
    _descriptors.clear();
    _tag_offsets.clear();
    _loops.clear();
    _edges.clear();
    _edge_defs.clear();
    _point_remap.clear();
  }

  /// Build loops, edges, canonicalize, and resolve crossings.
  /// `apply_to_face(tag, object, f)` calls `f(face)` with the face view.
  template <typename ApplyToFace, typename GetMeshPoint>
  auto
  build(const tf::intersect::tagged_intersections<Index, Int, 3> &intersections,
        const ApplyToFace &apply_to_face, const GetMeshPoint &get_mesh_point,
        tf::intersect_mode mode,
        const tf::exact::predicate_kernel<Int> &kernel = {}) -> void {
    clear();

    auto tag_offs = intersections.tag_offsets();
    _tag_offsets.allocate(tag_offs.size());
    tf::parallel_copy(tag_offs, tf::make_range(_tag_offsets));

    const auto &subranges = intersections.intersections();
    if (subranges.size() == 0)
      return;
    const auto &ipts = intersections.intersection_points();
    auto get_point = [&](Index tag, Index id) -> tf::point<Int, 3> {
      if (tag == -1)
        return ipts[id];
      return get_mesh_point(tag, id);
    };
    auto get_flat_id = [&](const auto &rec) -> Index {
      return intersections.get_flat_index(rec);
    };
    build_loops(subranges, apply_to_face, get_point, get_flat_id);
    build_edges(subranges, apply_to_face);
    intersect::graph::canonicalize_edges(_edge_defs, _edges);
    if (!(mode & tf::intersect_mode::resolve_self_crossing_contours) &&
        _tag_offsets.size() < (3 + 1)) {
      _points.allocate(ipts.size());
      tf::parallel_copy(ipts, tf::make_range(_points));
      intersect::graph::strip_base_loop_edges(_edge_defs.data_buffer(), _edges);
      intersect::graph::canonicalize_edges(_edge_defs, _edges);
    } else
      detect_and_split_crossings(ipts, apply_to_face, get_point, mode, kernel);
  }

private:
  template <typename Subranges, typename ApplyToFace, typename GetPoint,
            typename GetFlatId>
  auto build_loops(const Subranges &subranges, const ApplyToFace &apply_to_face,
                   const GetPoint &get_point, const GetFlatId &get_flat_id)
      -> void {
    auto n = subranges.size();
    _loops.offsets_buffer().allocate(n + 1);
    _loops.offsets_buffer()[0] = 0;

    std::size_t loop_i = 1;

    struct local_t {
      tf::buffer<Index> sizes;
      tf::buffer<intersect::graph::vertex<Index>> dirty;
      tf::buffer<intersect::graph::vertex<Index>> data;
      tf::buffer<intersect::graph::loop_node<Index, Int>> work;
      tf::buffer<intersect::graph::face_descriptor<Index>> descs;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_point_f = get_point;
      auto get_flat_id_f = get_flat_id;
      local.sizes.allocate(range.size());
      auto sit = local.sizes.begin();
      for (const auto &subrange : range) {
        auto old_size = local.data.size();
        auto tag = subrange[0].tag;
        auto object = subrange[0].object;
        local.dirty.clear();
        apply_to_face_f(tag, object, [&](const auto &face) {
          intersect::graph::extract_loop(subrange, face, tag, get_point_f,
                                         get_flat_id_f, local.work,
                                         local.dirty);
        });
        intersect::graph::clean_loop<Index>(local.dirty, local.data);
        local.descs.push_back({tag, object});
        *sit++ = static_cast<Index>(local.data.size() - old_size);
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.data, _loops.data_buffer());
      tf::core::append(local.descs, _descriptors);
      for (auto sz : local.sizes) {
        _loops.offsets_buffer()[loop_i] =
            _loops.offsets_buffer()[loop_i - 1] + sz;
        ++loop_i;
      }
    };

    tf::blocked_reduce_sequenced_aggregate(subranges, tf::none, local_t{}, task,
                                           agg);
  }

  template <typename Subranges, typename ApplyToFace>
  auto build_edges(const Subranges &subranges, const ApplyToFace &apply_to_face)
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
      auto apply_to_face_f = apply_to_face;
      local.counts.allocate(range.size());
      auto cit = local.counts.begin();
      for (auto &&[this_loop_idx, subrange] : range) {
        auto old_size = local.data.size();
        if (subrange.size() != 0) {
          auto tag = subrange[0].tag;
          auto object = subrange[0].object;
          apply_to_face_f(tag, object,
                          [&, &subrange = subrange,
                           &this_loop_idx = this_loop_idx](const auto &face) {
                            auto face_size = face.size();
                            local.extractor.extract(
                                subrange, face_size, this_loop_idx, all_loops,
                                subranges, apply_to_face_f, local.data);
                          });
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

    tf::blocked_reduce_sequenced_aggregate(tf::enumerate(subranges), tf::none,
                                           local_t{}, task, agg);
  }

  template <typename IPoints, typename ApplyToFace, typename GetPoint>
  auto detect_and_split_crossings(
      const IPoints &ipts, const ApplyToFace &apply_to_face,
      const GetPoint &get_point, tf::intersect_mode mode,
      const tf::exact::predicate_kernel<Int> &kernel) -> void {
    if (_edges.size() == 0) {
      _points.allocate(ipts.size());
      tf::parallel_copy(ipts, tf::make_range(_points));
      return;
    }

    auto n_ipts = static_cast<Index>(ipts.size());

    auto records = intersect::graph::gather_crossing_records<Index>(
        _edges, _edge_defs, apply_to_face, get_point, mode, kernel);
    if (records.size() == 0) {
      _points.allocate(n_ipts);
      tf::parallel_copy(ipts, tf::make_range(_points));
      intersect::graph::strip_base_loop_edges(_edge_defs.data_buffer(), _edges);
      intersect::graph::canonicalize_edges(_edge_defs, _edges);
      return;
    }

    intersect::graph::sort_crossing_records<Index>(records);
    auto [ee_range, ve_range, vv_range] =
        intersect::graph::find_type_ranges<Index>(records);

    tf::buffer<tf::point<Int, 3>> crossing_points;
    auto ee_offsets = intersect::graph::compute_ee_crossing_points<Index>(
        ee_range, _edge_defs, get_point, crossing_points);
    auto crossing_base = n_ipts;
    ve_range = intersect::graph::dedup_ve(ve_range);

    tf::buffer<std::array<Index, 2>> merge_pairs;
    intersect::graph::collect_vv_pairs<Index>(vv_range, merge_pairs);

    for (const auto &p : merge_pairs) {
      _crossing_point_ids.push_back(p[0]);
      _crossing_point_ids.push_back(p[1]);
    }

    auto entries = intersect::graph::collect_split_entries<Index>(
        ee_range, ee_offsets, ve_range, crossing_base);
    if (entries.size() > 0) {
      intersect::graph::split_edges(entries, crossing_base, crossing_points,
                                    _edge_defs.data_buffer(), _edges, _loops,
                                    get_point, merge_pairs);
    } else {
      intersect::graph::strip_base_loop_edges(_edge_defs.data_buffer(), _edges);
    }

    auto total_size =
        crossing_base + static_cast<Index>(crossing_points.size());

    for (Index i = crossing_base; i < total_size; ++i)
      _crossing_point_ids.push_back(i);

    if (merge_pairs.size() == 0) {
      _points.allocate(total_size);
      tf::parallel_copy(ipts, tf::take(_points, crossing_base));
      tf::parallel_copy(crossing_points, tf::drop(_points, crossing_base));
      intersect::graph::canonicalize_edges(_edge_defs, _edges);
      return;
    }

    _point_remap.allocate(total_size);
    auto n_classes =
        tf::make_dense_equivalence_class_map(merge_pairs, _point_remap);

    auto all_coords = tf::make_mapped_range(
        tf::make_sequence_range(total_size), [&](Index i) -> tf::point<Int, 3> {
          return i < crossing_base ? ipts[i]
                                   : crossing_points[i - crossing_base];
        });
    _points.allocate(n_classes);
    tf::parallel_copy(all_coords,
                      tf::make_indirect_range(_point_remap, _points));

    tf::parallel_for_each(_edge_defs.data_buffer(), [&](auto &e) {
      e.point_0 = _point_remap[e.point_0];
      e.point_1 = _point_remap[e.point_1];
    });
    tf::parallel_for_each(_loops.data_buffer(), [&](auto &v) {
      if (v.source == intersect::graph::vertex_source::created)
        v.id = _point_remap[v.id];
    });
    intersect::graph::clean_loops(_loops);

    intersect::graph::canonicalize_edges(_edge_defs, _edges);

    collect_crossing_point_ids();
  }

  auto collect_crossing_point_ids() -> void {
    if (_crossing_point_ids.size() == 0)
      return;
    if (_point_remap.size() > 0)
      for (auto &id : _crossing_point_ids)
        id = _point_remap[id];
    std::sort(_crossing_point_ids.begin(), _crossing_point_ids.end());
    _crossing_point_ids.erase_till_end(
        std::unique(_crossing_point_ids.begin(), _crossing_point_ids.end()));
  }

  tf::points_buffer<Int, 3> _points;
  tf::buffer<Index> _crossing_point_ids;
  tf::buffer<intersect::graph::face_descriptor<Index>> _descriptors;
  tf::buffer<Index> _tag_offsets;
  tf::offset_block_buffer<Index, intersect::graph::vertex<Index>> _loops;
  tf::offset_block_buffer<Index, Index> _edges;
  tf::offset_block_buffer<Index, intersect::graph::edge<Index>> _edge_defs;
  tf::buffer<Index> _point_remap;
};

} // namespace tf
