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
#include "./vertex.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>

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
    _snip_merges.clear();
    _snip_dropped.clear();
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
    const bool resolving =
        (mode & tf::intersect_mode::resolve_self_crossing_contours) ||
        _tag_offsets.size() >= (3 + 1);
    if (resolving && kernel.is_tolerated())
      _include_boundary_chords();
    intersect::graph::canonicalize_edges(_edge_defs, _edges);
    if (!resolving) {
      _materialize_plain(ipts);
    } else
      detect_and_split_crossings(ipts, apply_to_face, get_point, mode, kernel);
    _finalize_edges();
  }

  /// Tolerance welds insert created vertices into base loops whose
  /// flanking chords are otherwise invisible to crossing detection (no
  /// record segment spans them). Emit every created–created consecutive
  /// loop chord not already present as a stamped boundary edge, so the
  /// crossing pass tests the emitted boundary like any other segment.
  /// Both faces adjacent to the original edge carry identical welded
  /// vertices (edge-targeted records are duplicated), so the two chord
  /// instances canonicalize into one group and splits stay symmetric.
  /// Boundary-coincident edges are compacted back into the loops by
  /// _finalize_edges, exactly as for walk-emitted boundary sub-edges.
  auto _include_boundary_chords() -> void {
    auto &defs_raw = _edge_defs.data_buffer();
    auto &inst_raw = _edges.data_buffer();
    auto &offs = _edges.offsets_buffer();
    const auto n_faces = offs.size() - 1;

    tf::buffer<intersect::graph::edge<Index>> extra;
    tf::buffer<Index> extra_counts;
    extra_counts.allocate(n_faces);
    tf::buffer<std::array<Index, 2>> present;

    for (std::size_t f = 0; f < n_faces; ++f) {
      const auto begin = std::size_t(offs[f]);
      const auto end = std::size_t(offs[f + 1]);
      const auto old_extra = extra.size();
      auto &&loop = _loops[f];
      const auto m = loop.size();
      if (m != 0 && end != begin) {
        present.clear();
        for (std::size_t e = begin; e < end; ++e) {
          const auto &d = defs_raw[std::size_t(inst_raw[e])];
          present.push_back({std::min(d.point_0, d.point_1),
                             std::max(d.point_0, d.point_1)});
        }
        std::sort(present.begin(), present.end());
        const auto tag = _descriptors[f].tag;
        const auto object = _descriptors[f].object;
        for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
          const auto &a = loop[j];
          const auto &b = loop[i];
          if (a.source != intersect::graph::vertex_source::created ||
              b.source != intersect::graph::vertex_source::created ||
              a.id == b.id)
            continue;
          std::array<Index, 2> key{std::min(a.id, b.id),
                                   std::max(a.id, b.id)};
          if (std::binary_search(present.begin(), present.end(), key))
            continue;
          extra.push_back({short(tag), short(tag), object, object, a.id,
                           b.id, Index(0), static_cast<std::int16_t>(j),
                           std::int16_t(0)});
        }
      }
      extra_counts[f] = static_cast<Index>(extra.size() - old_extra);
    }
    if (extra.size() == 0)
      return;

    tf::buffer<intersect::graph::edge<Index>> new_defs;
    tf::buffer<Index> new_inst;
    tf::buffer<Index> new_offs;
    new_defs.allocate(defs_raw.size() + extra.size());
    new_inst.allocate(inst_raw.size() + extra.size());
    new_offs.allocate(offs.size());
    new_offs[0] = 0;
    std::size_t w = 0;
    std::size_t x = 0;
    for (std::size_t f = 0; f < n_faces; ++f) {
      for (std::size_t e = std::size_t(offs[f]); e < std::size_t(offs[f + 1]);
           ++e)
        new_defs[w++] = defs_raw[std::size_t(inst_raw[e])];
      for (Index k = 0; k < extra_counts[f]; ++k)
        new_defs[w++] = extra[x++];
      new_offs[f + 1] = static_cast<Index>(w);
    }
    for (std::size_t e = 0; e < new_defs.size(); ++e) {
      new_defs[e].id = static_cast<Index>(e);
      new_inst[e] = static_cast<Index>(e);
    }
    defs_raw = std::move(new_defs);
    inst_raw = std::move(new_inst);
    offs = std::move(new_offs);
  }

  /// The single finalize: loops cleaned once, ordinals derived from the
  /// final loops, boundary-coincident edges compacted away once, canonical
  /// ids assigned once. Runs at the end of every build path.
  auto _finalize_edges() -> void {
    intersect::graph::clean_loops(_loops);
    _restamp_boundary_ordinals();
    // Dead edges compact away: boundary-coincident pieces (ordinal
    // stamped) and zero-length edges — point-class merges can fuse an
    // edge's endpoints after emission, and a one-point edge cuts
    // nothing.
    bool any_dead = false;
    for (const auto &e : _edge_defs.data_buffer())
      if (e.ordinal != -1 || e.point_0 == e.point_1) {
        any_dead = true;
        break;
      }
    if (any_dead) {
      tf::buffer<intersect::graph::edge<Index>> new_edge_data;
      tf::buffer<Index> new_offsets;
      new_offsets.allocate(_edges.offsets_buffer().size());
      new_offsets[0] = 0;
      std::size_t fi = 1;
      for (auto face_edges : tf::make_range(_edges)) {
        for (auto inst : face_edges) {
          const auto &e = _edge_defs.data_buffer()[inst];
          if (e.ordinal == -1 && e.point_0 != e.point_1)
            new_edge_data.push_back(e);
        }
        new_offsets[fi++] = static_cast<Index>(new_edge_data.size());
      }
      _edge_defs.data_buffer() = std::move(new_edge_data);
      _edge_defs.offsets_buffer() = new_offsets;
      _edges.offsets_buffer() = std::move(new_offsets);
      _edges.data_buffer().allocate(_edge_defs.data_buffer().size());
      tf::parallel_iota(_edges.data_buffer(), 0);
    }
    intersect::graph::canonicalize_edges(_edge_defs, _edges);
  }

  /// Ordinals derived from the final loops. Point-class merges can move
  /// a vertex onto the boundary after emission and splitting, turning an
  /// interior chord piece into a duplicate of a base-loop segment; every
  /// earlier stamp ran before identity was final. An edge whose endpoints
  /// are consecutive in its face's loop is boundary (stamped with the
  /// loop position, flipped to the loop's direction); anything else is
  /// interior.
  auto _restamp_boundary_ordinals() -> void {
    auto loops = tf::make_range(_loops);
    auto edge_blocks = tf::make_range(_edges);
    struct seg_t {
      Index a, b;
      std::int16_t ord;
      Index want0;
    };
    struct local_t {
      tf::small_vector<seg_t, 16> segs;
    };
    tf::parallel_for_each(
        tf::make_sequence_range(loops.size()),
        [&](std::size_t li, local_t &local) {
          auto face_edges = edge_blocks[li];
          if (face_edges.size() == 0)
            return;
          auto loop = loops[li];
          const std::size_t n = loop.size();
          auto &segs = local.segs;
          segs.clear();
          for (std::size_t i = 0; i < n; ++i) {
            const auto &va = loop[i];
            const auto &vb = loop[(i + 1) % n];
            if (va.source != intersect::graph::vertex_source::created ||
                vb.source != intersect::graph::vertex_source::created)
              continue;
            if (va.id == vb.id)
              continue;
            segs.push_back({std::min(va.id, vb.id), std::max(va.id, vb.id),
                            std::int16_t(i), va.id});
          }
          std::sort(segs.begin(), segs.end(),
                    [](const seg_t &x, const seg_t &y) {
                      return std::make_pair(x.a, x.b) <
                             std::make_pair(y.a, y.b);
                    });
          for (auto inst : face_edges) {
            auto &e = _edge_defs.data_buffer()[inst];
            const Index a = std::min(e.point_0, e.point_1);
            const Index b = std::max(e.point_0, e.point_1);
            auto it = std::lower_bound(
                segs.begin(), segs.end(), std::make_pair(a, b),
                [](const seg_t &sg, const std::pair<Index, Index> &k) {
                  return std::make_pair(sg.a, sg.b) < k;
                });
            if (it == segs.end() || it->a != a || it->b != b) {
              e.ordinal = -1;
              e.sub_ordinal = -1;
            } else {
              e.ordinal = it->ord;
              e.sub_ordinal = 0;
              if (e.point_0 != it->want0)
                std::swap(e.point_0, e.point_1);
            }
          }
        },
        local_t{});
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
      tf::buffer<std::array<Index, 2>> snips;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_point_f = get_point;
      auto get_flat_id_f = get_flat_id;
      local.sizes.allocate(range.size());
      auto sit = local.sizes.begin();
      for (const auto &subrange : range) {
        auto old_size = local.data.size();
        Index tag = subrange[0].tag;
        auto object = subrange[0].object;
        local.dirty.clear();
        apply_to_face_f(tag, object, [&](const auto &face) {
          intersect::graph::extract_loop(subrange, face, tag, get_point_f,
                                         get_flat_id_f, local.work,
                                         local.dirty);
        });
        intersect::graph::clean_loop<Index>(local.dirty, local.data,
                                            &local.snips);
        local.descs.push_back({tag, object});
        *sit++ = static_cast<Index>(local.data.size() - old_size);
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.data, _loops.data_buffer());
      tf::core::append(local.descs, _descriptors);
      tf::core::append(local.snips, _snip_merges);
      for (auto sz : local.sizes) {
        _loops.offsets_buffer()[loop_i] =
            _loops.offsets_buffer()[loop_i - 1] + sz;
        ++loop_i;
      }
    };

    tf::blocked_reduce_sequenced_aggregate(subranges, tf::none, local_t{}, task,
                                           agg);

    if (_snip_merges.size() != 0) {
      _snip_dropped.reserve(_snip_merges.size());
      for (const auto &p : _snip_merges)
        _snip_dropped.push_back(p[0]);
      tbb::parallel_sort(_snip_dropped.begin(), _snip_dropped.end());
      _snip_dropped.erase_till_end(
          std::unique(_snip_dropped.begin(), _snip_dropped.end()));
      for (auto &p : _snip_merges)
        if (p[1] < p[0])
          std::swap(p[0], p[1]);
      tbb::parallel_sort(_snip_merges.begin(), _snip_merges.end());
      _snip_merges.erase_till_end(
          std::unique(_snip_merges.begin(), _snip_merges.end()));
    }
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
      _materialize_plain(ipts);
      return;
    }

    auto n_ipts = static_cast<Index>(ipts.size());

    auto records = intersect::graph::gather_crossing_records<Index>(
        _edges, _edge_defs, apply_to_face, get_point, mode, kernel);
    if (records.size() == 0) {
      _materialize_plain(ipts);
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

    // Snip merges join the same machinery; they are not crossings, so
    // they stay out of _crossing_point_ids.
    tf::core::append(_snip_merges, merge_pairs);

    auto entries = intersect::graph::collect_split_entries<Index>(
        ee_range, ee_offsets, ve_range, crossing_base);
    if (entries.size() > 0) {
      intersect::graph::split_edges(entries, crossing_base, crossing_points,
                                    _edge_defs.data_buffer(), _edges, _loops,
                                    get_point, merge_pairs);
    }

    auto total_size =
        crossing_base + static_cast<Index>(crossing_points.size());

    for (Index i = crossing_base; i < total_size; ++i)
      _crossing_point_ids.push_back(i);

    // Materialise points, remapping through merge classes only when there
    // are merges.
    if (merge_pairs.size() == 0) {
      _points.allocate(total_size);
      tf::parallel_copy(ipts, tf::take(_points, crossing_base));
      tf::parallel_copy(crossing_points, tf::drop(_points, crossing_base));
    } else {
      _point_remap.allocate(total_size);
      auto n_classes =
          tf::make_dense_equivalence_class_map(merge_pairs, _point_remap);

      auto all_coords = tf::make_mapped_range(
          tf::make_sequence_range(total_size),
          [&](Index i) -> tf::point<Int, 3> {
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
      _write_snip_survivor_coords(
          [&](Index i) -> tf::point<Int, 3> { return ipts[i]; });
    }

    collect_crossing_point_ids();
  }

  /// Deterministic class coordinates for snip merges: unlike VV merges,
  /// the merged points are NOT coincident, so the racy indirect copy must
  /// be overridden with the surviving point's position.
  template <typename Coords>
  auto _write_snip_survivor_coords(const Coords &coords) -> void {
    auto dropped = [&](Index x) {
      return std::binary_search(_snip_dropped.begin(), _snip_dropped.end(), x);
    };
    auto pts = _points.points();
    for (const auto &p : _snip_merges) {
      // Chained snips (a pair's survivor dropped by another loop's snip)
      // must not write a dropped tip's coordinates: only the pair holding
      // the class's true survivor writes. When two survivors merge through
      // a shared tip, the sorted pair order makes the last write the
      // deterministic pick.
      Index s;
      if (!dropped(p[0]))
        s = p[0];
      else if (!dropped(p[1]))
        s = p[1];
      else
        continue;
      pts[_point_remap[s]] = coords(s);
    }
  }

  /// Points materialization for build paths that skip crossing detection:
  /// snip merges must still collapse dropped ids or record references
  /// dangle.
  template <typename IPoints>
  auto _materialize_plain(const IPoints &ipts) -> void {
    if (_snip_merges.size() == 0) {
      _points.allocate(ipts.size());
      tf::parallel_copy(ipts, tf::make_range(_points));
      return;
    }
    _point_remap.allocate(ipts.size());
    auto n_classes =
        tf::make_dense_equivalence_class_map(_snip_merges, _point_remap);
    _points.allocate(n_classes);
    tf::parallel_copy(ipts, tf::make_indirect_range(_point_remap, _points));
    _write_snip_survivor_coords(
        [&](Index i) -> tf::point<Int, 3> { return ipts[i]; });
    tf::parallel_for_each(_edge_defs.data_buffer(), [&](auto &e) {
      e.point_0 = _point_remap[e.point_0];
      e.point_1 = _point_remap[e.point_1];
    });
    tf::parallel_for_each(_loops.data_buffer(), [&](auto &v) {
      if (v.source == intersect::graph::vertex_source::created)
        v.id = _point_remap[v.id];
    });
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
  // clean_loop antenna snips: (min, max) merge pairs seeded into the VV
  // point-merge, plus the dropped side so the class keeps the survivor's
  // coordinates. A snipped point stays referenced by records; without the
  // merge those references dangle.
  tf::buffer<std::array<Index, 2>> _snip_merges;
  tf::buffer<Index> _snip_dropped;
};

} // namespace tf
