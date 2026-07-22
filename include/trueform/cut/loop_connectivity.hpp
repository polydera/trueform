/*
 * Copyright (c) 2026 XLAB
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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/range.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/zip.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../topology/face_membership.hpp"
#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../topology/structures/compute_face_link_per_edge.hpp"
#include "./face_cuts.hpp"
#include "./face_regions.hpp"

#include <algorithm>
#include <numeric>

namespace tf {

/// @ingroup cut
/// @brief Per-loop per-edge connectivity of an arrangement's cut
///        loops: for each loop, per edge, the neighbouring loops
///        sharing that edge. Coplanar-dead loops (the arrangement
///        graph's stack scan decides them) are cleaned out on both
///        sides — their entries are empty and no live loop references
///        them.
///
/// Classification-tier: built and owned by
/// @ref tf::cut::component_labels. Detection of coplanar stacks does
/// NOT happen here — the dead mask is an input.
///
/// Retains no references to the `intersection_graph` or `face_cuts`.
///
/// @tparam Index The integer index type for vertex / face / loop ids.
template <typename Index> class loop_connectivity {
public:
  loop_connectivity() = default;

  /// @brief Number of input forms (tags).
  auto n_tags() const -> Index {
    return static_cast<Index>(_tag_loop_offsets.size()) - 1;
  }

  auto connectivity_per_face_edge() const {
    return tf::make_offset_block_range(
        _loop_edge_offsets,
        tf::make_offset_block_range(_edge_neighbour_offsets,
                                    tf::make_range(_neighbours)));
  }

  auto clear() -> void {
    _tag_loop_offsets.clear();
    _loop_edge_offsets.clear();
    _edge_neighbour_offsets.clear();
    _neighbours.clear();
  }

  /// @brief Populate the connectivity from an intersection graph and
  ///        face cuts, cleaning the given coplanar-dead loops out.
  ///
  /// Detection is NOT performed here — the dead mask comes from the
  /// arrangement graph's stack scan (one detection, read everywhere).
  /// `point_counts[t]` = number of points of form `t` — original ids
  /// flatten densely as `per-tag base + id`, no remapping.
  template <typename Int, typename DeadRange, typename Counts>
  auto build(const tf::intersection_graph<Index, Int> &ig,
             const tf::face_cuts<Index, Int> &fc, const DeadRange &dead_loops,
             const Counts &point_counts) -> void {
    clear();

    _compute_loop_connectivity(ig, fc, point_counts);
    bool any_dead = false;
    for (auto d : dead_loops)
      if (d) {
        any_dead = true;
        break;
      }
    if (any_dead)
      _clean_connectivity(dead_loops);

    auto tag_offs = fc.tag_offsets();
    _tag_loop_offsets.allocate(tag_offs.size());
    tf::parallel_copy(tag_offs, tf::make_range(_tag_loop_offsets));
  }

  /// @brief Populate the connectivity at REGION grain from the raw
  ///        structure: every region contributes ALL its walks — the
  ///        boundary first, then its holes in `loop_holes()` order
  ///        (the contract consumers rely on to know each edge slot's
  ///        vertex pair) — so hole adjacency (a hole walk borders the
  ///        interior piece's boundary) falls out of the shared-edge
  ///        structure instead of being patched in. Regions are atomic:
  ///        interior triangulation edges do not exist here, which is
  ///        what keeps coincident CDT diagonals out of the whole
  ///        classification layer by construction.
  /// @overload
  template <typename Int, typename DeadRange, typename Counts>
  auto build(const tf::face_regions<Index, Int> &fr,
             const DeadRange &dead_regions, const Counts &point_counts,
             Index n_created) -> void {
    clear();

    _compute_region_connectivity(fr, point_counts, n_created);
    bool any_dead = false;
    for (auto d : dead_regions)
      if (d) {
        any_dead = true;
        break;
      }
    if (any_dead)
      _clean_connectivity(dead_regions);

    auto tag_offs = fr.tag_offsets();
    _tag_loop_offsets.allocate(tag_offs.size());
    tf::parallel_copy(tag_offs, tf::make_range(_tag_loop_offsets));
  }

private:
  // ====================================================================
  // Private build helpers.
  // ====================================================================

  /// @brief Build per-loop per-edge connectivity from `ig` + `fc`.
  ///
  /// Populates @ref _loop_edge_offsets, @ref _edge_neighbour_offsets,
  /// and @ref _neighbours. At this point the connectivity is the
  /// *full* graph; coplanar cleanup filters it afterwards.
  ///
  /// Flat vertex ids: created ids stay; original ids lift densely by
  /// a per-tag base of the form's point count — no per-vertex
  /// remapping (measured faster than hash maps on bunny-pair and the
  /// geological corpus).
  template <typename Int, typename Counts>
  auto _compute_loop_connectivity(const tf::intersection_graph<Index, Int> &ig,
                                  const tf::face_cuts<Index, Int> &fc,
                                  const Counts &point_counts) -> void {
    using source = tf::intersect::graph::vertex_source;

    const auto &src = fc.loops_buffer();
    if (!src.size()) {
      _loop_edge_offsets.clear();
      _edge_neighbour_offsets.clear();
      _neighbours.clear();
      return;
    }

    auto n_ipts = static_cast<Index>(ig.points().size());
    auto tag_offs = fc.tag_offsets();
    auto n_tags = tag_offs.size() - 1;

    tf::buffer<Index> tag_base;
    tag_base.allocate(n_tags + 1);
    tag_base[0] = n_ipts;
    for (std::size_t t = 0; t < n_tags; ++t)
      tag_base[t + 1] = tag_base[t] + static_cast<Index>(point_counts[t]);

    tf::buffer<Index> flat_data;
    flat_data.allocate(src.data_buffer().size());
    auto tag_src = tf::make_offset_block_range(
        tag_offs,
        tf::make_offset_block_range(src.offsets_buffer(), src.data_buffer()));
    auto tag_dst = tf::make_offset_block_range(
        tag_offs, tf::make_offset_block_range(src.offsets_buffer(), flat_data));
    tf::parallel_for_each(
        tf::enumerate(tf::zip(tag_src, tag_dst)), [&](auto pair) {
          auto &&[tag, tup] = pair;
          auto &&[src_loops, dst_loops] = tup;
          auto base = tag_base[tag];
          for (auto &&[s, d] : tf::zip(src_loops, dst_loops))
            for (std::size_t i = 0; i < s.size(); ++i)
              d[i] =
                  (s[i].source == source::created) ? s[i].id : base + s[i].id;
        });

    Index n_flat = tag_base[n_tags];

    auto flat_loops = tf::make_offset_block_range(src.offsets_buffer(),
                                                  tf::make_range(flat_data));
    tf::face_membership<Index> fm;
    fm.build(tf::make_faces(flat_loops), n_flat, flat_data.size());

    tf::topology::compute_face_link_per_edge(
        flat_loops, fm, _edge_neighbour_offsets, _neighbours);

    // per-loop offsets are exactly the loop-vertex offsets (K vertices
    // = K edges)
    _loop_edge_offsets.allocate(src.offsets_buffer().size());
    tf::parallel_copy(tf::make_range(src.offsets_buffer()),
                      tf::make_range(_loop_edge_offsets));
  }


  /// Region-grain build, the loop-grain algorithm verbatim at region
  /// grain: vertex -> REGION membership (each region one face — its
  /// walks' vertices concatenated boundary-first), then the per-edge
  /// link by sorted-intersection of the two endpoints' memberships
  /// (self-excluding, multiplicity-correct for pinches/slits). The
  /// candidate edge test wraps PER WALK — a region is not one cycle —
  /// so walk closing edges count and cross-walk seams do not exist.
  /// Neighbour ids are region ids by construction: no remap pass.
  template <typename Int, typename Counts>
  auto _compute_region_connectivity(const tf::face_regions<Index, Int> &fr,
                                    const Counts &point_counts,
                                    Index n_created) -> void {
    using source = tf::intersect::graph::vertex_source;

    auto loops = fr.loops();
    auto holes = fr.holes();
    auto loop_holes = fr.loop_holes();
    auto descs = fr.descriptors();
    const std::size_t n_regions = loops.size();
    if (!n_regions) {
      _loop_edge_offsets.clear();
      _edge_neighbour_offsets.clear();
      _neighbours.clear();
      return;
    }

    const std::size_t n_tags = fr.tag_offsets().size() - 1;
    tf::buffer<Index> tag_base;
    tag_base.allocate(n_tags + 1);
    tag_base[0] = n_created;
    for (std::size_t t = 0; t < n_tags; ++t)
      tag_base[t + 1] = tag_base[t] + static_cast<Index>(point_counts[t]);

    _loop_edge_offsets.allocate(n_regions + 1);
    _loop_edge_offsets[0] = 0;
    for (std::size_t li = 0; li < n_regions; ++li) {
      std::size_t cnt = loops[Index(li)].size();
      for (auto h : loop_holes[Index(li)])
        cnt += holes[h].size();
      _loop_edge_offsets[li + 1] = _loop_edge_offsets[li] + Index(cnt);
    }

    tf::buffer<Index> flat_data;
    flat_data.allocate(std::size_t(_loop_edge_offsets[n_regions]));
    tf::parallel_for_each(
        tf::make_sequence_range(n_regions), [&](std::size_t li) {
          const Index base = tag_base[std::size_t(descs[li].tag)];
          Index off = _loop_edge_offsets[li];
          auto put = [&](const auto &walk) {
            for (std::size_t i = 0; i < walk.size(); ++i)
              flat_data[std::size_t(off++)] =
                  walk[i].source == source::created ? walk[i].id
                                                    : base + walk[i].id;
          };
          put(loops[Index(li)]);
          for (auto h : loop_holes[Index(li)])
            put(holes[h]);
        });

    const Index n_flat = tag_base[n_tags];
    auto region_faces = tf::make_offset_block_range(
        tf::make_range(_loop_edge_offsets), tf::make_range(flat_data));
    tf::face_membership<Index> fm;
    fm.build(tf::make_faces(region_faces), n_flat, flat_data.size());

    // Per-walk wrap over a candidate's flat slice: how many times does
    // it hold the undirected edge (v0, v1)? `first_only` early-exits
    // at one — the cheap existence probe of the loop-grain algorithm.
    auto edge_count_in = [&](Index cand, Index v0, Index v1,
                             bool first_only) -> Index {
      Index c = 0;
      Index off = _loop_edge_offsets[std::size_t(cand)];
      auto scan = [&](Index n) {
        Index p = n - 1;
        for (Index i = 0; i < n; p = i++) {
          const Index a = flat_data[std::size_t(off + p)];
          const Index b = flat_data[std::size_t(off + i)];
          c += Index((char(a == v0) & char(b == v1)) |
                     (char(a == v1) & char(b == v0)));
          if (first_only && c)
            return;
        }
        off += n;
      };
      scan(Index(loops[cand].size()));
      if (first_only && c)
        return c;
      for (auto h : loop_holes[cand]) {
        scan(Index(holes[h].size()));
        if (first_only && c)
          return c;
      }
      return c;
    };

    struct local_t {
      tf::buffer<Index> offsets;
      tf::buffer<Index> ids;
    };
    auto task = [&](auto &&range, local_t &local) {
      for (const auto &[li_z, region] : range) {
        const Index li = static_cast<Index>(li_z);
        const Index base = _loop_edge_offsets[std::size_t(li)];
        auto do_edge = [&](Index v0, Index v1) {
          local.offsets.push_back(Index(local.ids.size()));
          const auto &m0 = fm[v0];
          const auto &m1 = fm[v1];
          auto it0 = m0.begin();
          auto end0 = m0.end();
          auto it1 = m1.begin();
          auto end1 = m1.end();
          // Sorted-range intersection; a region revisiting a vertex
          // appears repeatedly in both memberships, so a candidate can
          // match more than once — emit exactly its edge multiplicity
          // (slit = twice, lobe pinch = once), the loop-grain rule.
          Index prev = li;
          Index emitted = 0;
          Index count = 0;
          bool full_count = false;
          while ((it0 != end0) & (it1 != end1)) {
            if (*it0 > *it1)
              ++it0;
            else {
              if (char(Index(*it0) != li) & char(!(*it1 > *it0))) {
                const Index cand = Index(*it0);
                if (cand != prev) {
                  prev = cand;
                  emitted = 0;
                  full_count = false;
                  count = edge_count_in(cand, v0, v1, true);
                } else if (!full_count) {
                  full_count = true;
                  count = edge_count_in(cand, v0, v1, false);
                }
                if (emitted < count) {
                  ++emitted;
                  local.ids.push_back(cand);
                }
                ++it0;
              }
              ++it1;
            }
          }
        };
        Index off = base;
        auto do_walk = [&](Index n) {
          for (Index j = 0; j < n; ++j)
            do_edge(flat_data[std::size_t(off + j)],
                    flat_data[std::size_t(off + (j + 1) % n)]);
          off += n;
        };
        do_walk(Index(region.size()));
        for (auto h : loop_holes[li])
          do_walk(Index(holes[h].size()));
      }
    };
    auto aggregate = [](const local_t &local,
                        std::tuple<tf::buffer<Index> &, tf::buffer<Index> &>
                            result) {
      auto &[offsets, ids] = result;
      const auto old_size = ids.size();
      ids.reallocate(old_size + local.ids.size());
      std::copy(local.ids.begin(), local.ids.end(), ids.begin() + old_size);
      const auto old_offsets = offsets.size();
      offsets.reallocate(old_offsets + local.offsets.size());
      auto it = offsets.begin() + old_offsets;
      for (auto o : local.offsets)
        *it++ = o + Index(old_size);
    };
    _edge_neighbour_offsets.clear();
    _neighbours.clear();
    _edge_neighbour_offsets.reserve(std::size_t(_loop_edge_offsets[n_regions]) +
                                    1);
    tf::blocked_reduce_sequenced_aggregate(
        tf::enumerate(loops),
        std::tie(_edge_neighbour_offsets, _neighbours), local_t{}, task,
        aggregate);
    _edge_neighbour_offsets.push_back(Index(_neighbours.size()));
  }

  /// @brief Second pass: re-copy connectivity with marked (dead) loops
  ///        removed.
  ///
  /// Rebuilds @ref _loop_edge_offsets, @ref _edge_neighbour_offsets,
  /// and @ref _neighbours in place so that:
  ///   - dead loops have a zero-length per-loop edge range,
  ///   - no live loop's neighbour list references a dead loop.
  ///
  /// Loop ids stay aligned with `face_cuts.descriptors()` ordering.
  /// Edge slot ids do shift (dead loops contribute zero edges to the
  /// new edge array).
  ///
  /// @param dead_loops_mask Per-loop boolean; `true` means the loop
  ///        was marked as a coplanar duplicate during the dedup pass.
  ///        Size must equal `_loop_edge_offsets.size() - 1`.
  template <typename Mask>
  auto _clean_connectivity(const Mask &dead_loops_mask) -> void {
    const Index n_loops = static_cast<Index>(_loop_edge_offsets.size() - 1);
    if (n_loops == 0)
      return;

    // ---- 1. Per-loop kept edge count, written directly into the new
    //         offsets buffer at index l+1; then in-place prefix sum.
    tf::buffer<Index> new_loop_edge_offsets;
    new_loop_edge_offsets.allocate(static_cast<std::size_t>(n_loops + 1));
    new_loop_edge_offsets[0] = Index(0);
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      new_loop_edge_offsets[l + 1] =
          dead_loops_mask[l]
              ? Index(0)
              : Index(_loop_edge_offsets[l + 1] - _loop_edge_offsets[l]);
    });
    std::partial_sum(new_loop_edge_offsets.begin(), new_loop_edge_offsets.end(),
                     new_loop_edge_offsets.begin());
    const Index new_n_edges = new_loop_edge_offsets[n_loops];

    // ---- 2. Per kept edge: count live neighbours directly into the
    //         new edge-offsets buffer at index e+1; then in-place
    //         prefix sum.
    tf::buffer<Index> new_edge_neighbour_offsets;
    new_edge_neighbour_offsets.allocate(
        static_cast<std::size_t>(new_n_edges + 1));
    new_edge_neighbour_offsets[0] = Index(0);
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      if (dead_loops_mask[l])
        return;
      const Index old_first = _loop_edge_offsets[l];
      const Index old_last = _loop_edge_offsets[l + 1];
      Index new_edge_idx = new_loop_edge_offsets[l];
      for (Index e = old_first; e < old_last; ++e, ++new_edge_idx) {
        const Index nb_first = _edge_neighbour_offsets[e];
        const Index nb_last = _edge_neighbour_offsets[e + 1];
        Index live = 0;
        for (Index k = nb_first; k < nb_last; ++k)
          if (!dead_loops_mask[_neighbours[k]])
            ++live;
        new_edge_neighbour_offsets[new_edge_idx + 1] = live;
      }
    });
    std::partial_sum(new_edge_neighbour_offsets.begin(),
                     new_edge_neighbour_offsets.end(),
                     new_edge_neighbour_offsets.begin());
    const Index new_n_neighbours = new_edge_neighbour_offsets[new_n_edges];

    // ---- 3. Fill the live neighbours into the new flat buffer.
    tf::buffer<Index> new_neighbours;
    new_neighbours.allocate(static_cast<std::size_t>(new_n_neighbours));
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      if (dead_loops_mask[l])
        return;
      const Index old_first = _loop_edge_offsets[l];
      const Index old_last = _loop_edge_offsets[l + 1];
      Index new_edge_idx = new_loop_edge_offsets[l];
      for (Index e = old_first; e < old_last; ++e, ++new_edge_idx) {
        const Index nb_first = _edge_neighbour_offsets[e];
        const Index nb_last = _edge_neighbour_offsets[e + 1];
        Index write_pos = new_edge_neighbour_offsets[new_edge_idx];
        for (Index k = nb_first; k < nb_last; ++k) {
          const Index nb = _neighbours[k];
          if (!dead_loops_mask[nb])
            new_neighbours[write_pos++] = nb;
        }
      }
    });

    _loop_edge_offsets = std::move(new_loop_edge_offsets);
    _edge_neighbour_offsets = std::move(new_edge_neighbour_offsets);
    _neighbours = std::move(new_neighbours);
  }

  // ====================================================================
  // Private data.
  // ====================================================================

  // --------------------------------------------------------------------
  // Per-tag loop offsets: tag t's loops occupy
  //   [_tag_loop_offsets[t], _tag_loop_offsets[t+1]).
  // Size: n_tags + 1.
  // --------------------------------------------------------------------
  tf::buffer<Index> _tag_loop_offsets;

  // --------------------------------------------------------------------
  // Per-loop per-edge connectivity, cleaned of dead loops.
  //
  // Layout: three buffers form a two-level offset block.
  //   - _loop_edge_offsets has size n_loops + 1.
  //     For loop_id, its per-edge slot range is
  //     [_loop_edge_offsets[loop_id], _loop_edge_offsets[loop_id+1]).
  //     For a dead loop the range is empty.
  //   - _edge_neighbour_offsets indexes into _neighbours. For an edge
  //     slot e, neighbours live in [..[e], ..[e+1]).
  //   - _neighbours is the flat list of neighbour loop ids; every
  //     entry refers to a live loop.
  // --------------------------------------------------------------------
  tf::buffer<Index> _loop_edge_offsets;
  tf::buffer<Index> _edge_neighbour_offsets;
  tf::buffer<Index> _neighbours;
};

} // namespace tf
