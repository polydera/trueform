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

#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/slice.hpp"
#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/index_hash_map.hpp"
#include "../core/none.hpp"
#include "../core/points_buffer.hpp"
#include "../core/memory.hpp"
#include "../exact/edge_edge_closest_point.hpp"
#include "../exact/meta.hpp"
#include "../exact/projection_axes.hpp"
#include "../topology/cdt_refine_config.hpp"
#include "../topology/cdt_refiner.hpp"
#include "../topology/constrained_delaunay_triangulator.hpp"
#include "./face_regions.hpp"
#include "./loop_connectivity.hpp"
#include "./refine_sizing.hpp"
#include "./resolve_face_edge.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <type_traits>

namespace tf::cut {

/// Triangulates every region of a @ref tf::face_regions structure with
/// the constrained Delaunay triangulator: the boundary walk plus hole
/// walks enter as region-boundary constraints, interior is the
/// parity-odd side, winding follows the loop's projected orientation.
/// No ear cutting, no hole patching.
///
/// A refused loop enters the recovery loop: rediscover with
/// intersections allowed (a throwaway instrument), materialize each
/// crossing ONCE (an exact closest-approach point of the two parent
/// segments — deterministic for every carrier), broadcast the splits to
/// every loop carrying a parent edge, and retry preserving. Rounds
/// repeat until everything is clear; survivors of the round cap stay in
/// `failed` — the loud signal.
template <typename Index, typename Int = tf::exact::int32>
class region_triangulator {
  using vertex_t = intersect::graph::vertex<Index>;
  using desc_t = intersect::graph::face_descriptor<Index>;
  using source = intersect::graph::vertex_source;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  /// A physical edge as its two endpoint identities, min-first.
  using ekey_t = std::array<std::array<Index, 2>, 2>;

  static constexpr int k_max_rounds = 4;

public:
  /// Per loop `[begin, end)` into `tris`.
  tf::buffer<std::array<Index, 2>> ranges;
  /// Winding flip for folded (coplanar-dead) loops: their range aliases
  /// the survivor's block, and a consumer emits it with corners 1 <-> 2
  /// swapped when set — exactly like emit_stored_triangulations.
  tf::buffer<char> rev;
  tf::buffer<std::array<vertex_t, 3>> tris;
  /// Loops still refused after the recovery rounds.
  tf::buffer<Index> failed;
  Index n_failed = 0;
  /// Refiner refusals in the last refine, and how many the crossing
  /// recovery conformed.
  Index n_refine_refused = 0;
  Index n_refine_recovered = 0;
  /// Points materialized by recovery; created ids at or above
  /// `n_ig_created` index this table at `id - n_ig_created`.
  tf::buffer<tf::point<Int, 3>> extra_points;
  Index n_ig_created = 0;

  /// Splits recovery placed on ORIGINAL mesh edges, grouped per edge:
  /// `{tag, u, v}` with the inserted vertices in order from `u` to `v`.
  /// The integration stage needs exactly this to find the uncut faces
  /// that now carry a vertex — they must re-triangulate to conform and
  /// must no longer be emitted as untouched originals.
  struct original_edge_split {
    Index tag;
    Index u, v; // original vertex ids, u < v
    vertex_t vertex;
  };
  auto original_edge_splits() const {
    return tf::make_range(_original_splits);
  }

  /// Faces that were uncut until a split landed on one of their edges.
  /// They are triangulated like any region and their blocks continue
  /// `ranges` past the structure's loops, so a consumer walks one list:
  /// the structure's descriptors followed by these. A face with no
  /// descriptor anywhere is untouched — no separate "conforming"
  /// bookkeeping, and no way to emit one twice.
  auto promoted_descriptors() const { return tf::make_range(_promoted); }

  /// Tag count the identity keys are built over — `resolve_key` tags
  /// created identities with exactly this value.
  auto n_tags() const -> Index { return _n_tags; }

  /// Index into the promoted list for `(tag, object)`, or -1 — the
  /// emit-time lookup of the refined path's conforming stream: a
  /// promoted face keeps its surface classification and swaps only its
  /// geometry for the conforming triangle block. `_promoted` is
  /// (tag, object)-sorted by construction.
  auto promoted_index(Index tag, Index object) const -> Index {
    auto it = std::lower_bound(_promoted.begin(), _promoted.end(),
                               desc_t{tag, object}, [](const desc_t &a,
                                                       const desc_t &b) {
                                 return a.tag != b.tag ? a.tag < b.tag
                                                       : a.object < b.object;
                               });
    if (it == _promoted.end() || it->tag != tag || it->object != object)
      return Index(-1);
    return Index(it - _promoted.begin());
  }

  /// The exposed triangle block of promoted entry `pi` (pairs with
  /// @ref promoted_index).
  auto promoted_range(Index pi) const -> std::array<Index, 2> {
    const std::size_t n_struct = _exposed_ranges.size() - _promoted.size();
    return _exposed_ranges[n_struct + std::size_t(pi)];
  }

  /// Identities retired by welds: `from` no longer names a vertex, the
  /// survivor does. Consumers holding vertex identities from before the
  /// triangulation (uncut faces, curve paths) must resolve through this
  /// before comparing. Sorted by `from`, transitively closed.
  auto merges() const { return tf::make_range(_merge); }

  /// The @ref tf::face_cuts exposure at triangle grain: every exposed
  /// loop IS one triangle of the finalized stream — contiguous per tag
  /// across structure regions, dead coplanar members (their survivor's
  /// triangles re-emitted under the dead member's OWN descriptor, with
  /// corners 1 <-> 2 swapped when `rev`) and promoted faces. With no
  /// promoted faces and no applied aliases this is a view straight over
  /// `tris` — already that order by construction.
  auto loops() const {
    const auto &src = _exposed_owned ? _exposed_triangles : tris;
    return tf::make_range(src);
  }
  /// One descriptor per exposed triangle, from its owning region.
  auto descriptors() const { return tf::make_range(_exposed_descriptors); }
  /// Per-tag `[begin, end)` triangle offsets into `loops()`.
  auto tag_offsets() const { return tf::make_range(_tag_offsets); }
  /// Per exposed face `[begin, end)` triangle offsets into `loops()` —
  /// the triangles-per-face grouping (the analog of face_cuts'
  /// loops-per-face descriptor runs). A face with no exposed triangles
  /// carries no block, so every block is non-empty and its descriptor
  /// is `descriptors()[face_offsets()[f]]`.
  auto face_offsets() const { return tf::make_range(_face_offsets); }
  /// Per structure loop (and promoted face past them), its `[begin,
  /// end)` block in the exposed stream — the loop-grain to
  /// triangle-grain map, e.g. for re-expressing the graph's per-region
  /// coplanar stacks at the exposed grain.
  auto exposed_ranges() const { return tf::make_range(_exposed_ranges); }

  /// Cut-and-consumed faces of form `tag`, an owned copy of the
  /// structure's store: consumers deciding cut/uncut must treat them as
  /// cut, never as untouched originals.
  auto deleted(Index tag) const {
    return tf::slice(tf::make_range(_deleted),
                     std::size_t(_deleted_offsets[std::size_t(tag)]),
                     std::size_t(_deleted_offsets[std::size_t(tag) + 1]));
  }
  auto deleted_offsets() const { return tf::make_range(_deleted_offsets); }

  auto loop_tris(Index li) const {
    auto r = ranges[std::size_t(li)];
    return tf::make_range(tris.begin() + r[0], tris.begin() + r[1]);
  }

  auto clear() -> void {
    ranges.clear();
    rev.clear();
    tris.clear();
    failed.clear();
    n_failed = 0;
    extra_points.clear();
    n_ig_created = 0;
    _n_tags = 0;
    _splits.clear();
    _original_splits.clear();
    _promoted.clear();
    _merge.clear();
    _steiners.clear();
    _region_fingerprints.clear();
    _refined_quality = -1.f;
    _promoted_steiners.clear();
    _promoted_interior_of.clear();
    _promoted_interior_base = 0;
    n_refine_refused = 0;
    n_refine_recovered = 0;
    _aliased = false;
    _exposed_owned = false;
    _exposed_triangles.clear();
    _exposed_descriptors.clear();
    _exposed_ranges.clear();
    _face_offsets.clear();
    _tag_offsets.clear();
    _deleted.clear();
    _deleted_offsets.clear();
  }

  /// Quality refinement over the same machinery: the refiner proposes
  /// splits on the boundary, they enter the split table like any other,
  /// and the shared retry/promotion path re-triangulates every carrier.
  /// Called after the structure has settled, so refinement never has to
  /// reason about crossings.
  template <typename Forms, typename ApplyToFace, typename GetMeshPoint>
  auto refine(const tf::face_regions<Index, Int> &fr,
              const tf::intersection_graph<Index, Int> &ig, const Forms &forms,
              const ApplyToFace &apply_to_face,
              const GetMeshPoint &get_mesh_point,
              const tf::cdt_refine_config &config) -> void {
    const tf::buffer<char> no_dead;
    const tf::buffer<std::array<Index, 3>> no_pairs;
    auto apply_form = [&forms](Index t, auto &&f) { f(forms[t]); };
    refine_applied(fr, ig, Index(forms.size()), apply_form, apply_to_face,
                   get_mesh_point, config, tf::make_range(no_dead),
                   tf::make_range(no_pairs));
  }

  /// The refined pass with the graph's stack context: dead regions and
  /// coplanar triples re-assert their aliases after the re-splice
  /// (PLAN §8 — a dead region re-emits its survivor's refined block).
  template <typename ApplyToForm, typename ApplyToFace, typename GetMeshPoint,
            typename DeadLoops, typename CoplanarPairs>
  auto refine_applied(const tf::face_regions<Index, Int> &fr,
                      const tf::intersection_graph<Index, Int> &ig,
                      Index n_tags, const ApplyToForm &apply_to_form,
                      const ApplyToFace &apply_to_face,
                      const GetMeshPoint &get_mesh_point,
                      const tf::cdt_refine_config &config,
                      const DeadLoops &dead_loops,
                      const CoplanarPairs &coplanar_pairs) -> void {
    auto descs = fr.descriptors();
    auto loops = fr.loops();
    auto loop_holes = fr.loop_holes();
    auto ipts = ig.points();
    scratch_t sc;
    tf::cdt_refiner<Index, Int, Int> rf;

    // The splittable mask reads the region connectivity — built
    // locally, as the reference build_refined built its own (the
    // arrangement tier carries only the coplanar stacks).
    tf::loop_connectivity<Index> conn_own;
    {
      tf::buffer<Index> point_counts;
      point_counts.allocate(std::size_t(n_tags));
      for (Index t = 0; t < n_tags; ++t)
        apply_to_form(t, [&](const auto &form) {
          point_counts[std::size_t(t)] =
              static_cast<Index>(form.points().size());
        });
      conn_own.build(fr, dead_loops, tf::make_range(point_counts),
                     Index(std::size_t(n_ig_created) + extra_points.size()));
    }
    auto conn = conn_own.connectivity_per_face_edge();

    // Intersection edges never receive quality splits. An edge is one
    // when the regions of its tag traverse it three or more times in
    // total — counting repeats by the same region, which the per-edge
    // neighbor connectivity cannot see (it only names OTHER regions,
    // so an edge a region walks twice looks ordinary there). Crossing
    // recovery is exempt: it must split whatever crosses.
    tf::buffer<std::array<std::array<Index, 2>, 2>> seam_keys;
    {
      tf::generic_generate(
          tf::enumerate(loops), seam_keys,
          [&](const auto &item, auto &out) {
            auto &&[li_z, loop_b] = item;
            const auto &loop = loop_b;
            const Index li = Index(li_z);
            const auto d = descs[li];
            if (d.tag == Index(-1) || loop.size() < 3 ||
                (dead_loops.size() != 0 && bool(dead_loops[li_z])))
              return;
            auto scan = [&](const auto &walk) {
              const std::size_t n = walk.size();
              for (std::size_t i = 0, j = n - 1; i < n; j = i++)
                out.push_back(
                    edge_key(id_of(d.tag, walk[j]), id_of(d.tag, walk[i])));
            };
            scan(loop);
            for (auto h : loop_holes[li])
              scan(fr.holes()[h]);
          });
      tbb::parallel_sort(seam_keys.begin(), seam_keys.end());
    }
    auto is_seam = [&](const std::array<std::array<Index, 2>, 2> &k) -> bool {
      auto r = std::equal_range(seam_keys.begin(), seam_keys.end(), k);
      return r.second - r.first >= 3;
    };

    // discovery: every live region proposes splits on its own walk.
    // Records only — the point each one names is materialized once,
    // after the proposals are merged, so threads never touch the
    // shared table.
    struct local_t {
      scratch_t sc;
      tf::cdt_refiner<Index, Int, Int> rf;
      tf::buffer<bool> splittable;
      tf::buffer<std::array<vertex_t, 3>> scrap;
      tf::buffer<proposal_t> proposals;
      tf::buffer<tf::point<Int, 3>> steiner;
      tf::buffer<std::array<Index, 2>> steiner_of; // (region, count)
      tf::buffer<char> usedp;
      tf::buffer<char> onc;
      tf::buffer<char> is_input;
      tf::buffer<Index> rf_failed;
      tf::buffer<rec_proposal_t> rec_proposals;
      tf::buffer<std::array<vertex_t, 2>> rec_parent_edges;
    };
    tf::buffer<proposal_t> proposals;
    tf::buffer<tf::point<Int, 3>> steiner_pool;
    tf::buffer<std::array<Index, 2>> steiner_of;
    tf::buffer<Index> rf_failed;
    tf::buffer<rec_proposal_t> rec_proposals;
    tf::buffer<std::array<vertex_t, 2>> rec_parent_edges;
    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (auto &&[li_z, d, loop, hole_ids] : range) {
        (void)li_z;
        // dead coplanar members follow their survivor's block; their
        // connectivity rows are zero-length by contract, so indexing
        // them here would read the next loop's slots
        if (d.tag == Index(-1) || loop.size() < 3 ||
            (dead_loops.size() != 0 && bool(dead_loops[li_z])))
          continue;
        // convergence gate: fixed point unless the bar was raised
        if (_region_fingerprints.size() != 0 &&
            config.min_quality <= _refined_quality &&
            _region_fingerprints[std::size_t(li_z)] ==
                region_fingerprint(fr, Index(li_z)))
          continue;
        local.scrap.clear();
        if (!run_loop(fr, d, loop, hole_ids, apply_to_face_f,
                      get_mesh_point_f, ipts, local.sc, local.scrap, true,
                      /*assemble_only=*/true)) {
          local.rf_failed.push_back(Index(li_z));
          continue;
        }
        inject_persisted_steiners(Index(li_z), local.sc);
        auto edges = tf::make_edges(tf::make_blocked_range<2>(
            tf::make_range(local.sc.cons.data(), local.sc.cons.size())));
        // The reference mask: whole border edges and single same-tag
        // cut neighbours may split; cross-tag seams and non-manifold
        // fans stay frozen. Sub-edges inherit their parent walk
        // slot's neighbour row; their proposals re-key onto the
        // parent edge below.
        local.splittable.allocate(edges.size());
        {
          auto row = conn[Index(li_z)];
          for (std::size_t e = 0; e < local.sc.cons_slot.size(); ++e) {
            auto nbs = row[local.sc.cons_slot[e]];
            // a split-born sub-edge takes its wholeness from the
            // parent walk edge: the fragment freeze targets cut-born
            // fragments, and the sub-edge's proposal re-keys onto the
            // parent, which the promoted neighbour conforms to
            const auto &cv = bool(local.sc.cons_sub[e])
                                 ? local.sc.cons_parent[e]
                                 : local.sc.cons_verts[e];
            const bool whole =
                cv[0].source ==
                    tf::intersect::graph::vertex_source::original &&
                cv[1].source == tf::intersect::graph::vertex_source::original;
            local.splittable[e] =
                ((nbs.size() == 0 && whole) ||
                 (nbs.size() == 1 && descs[nbs[0]].tag == d.tag)) &&
                !is_seam(edge_key(
                    id_of(d.tag, local.sc.cons_verts[e][0]),
                    id_of(d.tag, local.sc.cons_verts[e][1])));
          }
        }
        local.rf.clear();
        if (!local.rf.build(tf::make_points(local.sc.pts), edges,
                            tf::make_constant_range(true, edges.size()),
                            tf::make_range(local.splittable), config)) {
          // scratch is already assembled — discover crossings now,
          // retry after the round's serial materialization
          local.rf_failed.push_back(Index(li_z));
          discover_crossings(local.sc, d.tag, local.rec_proposals,
                             local.rec_parent_edges);
          continue;
        }
        auto recs = local.rf.constraint_splits();
        {
          const Index t_hi_v = Index(1) << k_param_bits;
          for (std::size_t e = 0; e < local.sc.cons_verts.size(); ++e) {
            if (recs[e].size() == 0)
              continue;
            // rebase onto the parent edge in the canonical frame, so
            // every carrier's full-edge query consumes the proposal
            const bool sub = bool(local.sc.cons_sub[e]);
            const auto &pedge =
                sub ? local.sc.cons_parent[e] : local.sc.cons_verts[e];
            Index ta, tb;
            if (sub) {
              ta = local.sc.cons_t[e][0];
              tb = local.sc.cons_t[e][1];
            } else {
              const bool fwd =
                  id_of(d.tag, pedge[0]) < id_of(d.tag, pedge[1]);
              ta = fwd ? Index(0) : t_hi_v;
              tb = t_hi_v - ta;
            }
            for (const auto &r : recs[e]) {
              const std::int64_t den = std::int64_t(1) << r.depth;
              const std::int64_t x = (std::int64_t(tb) - std::int64_t(ta)) *
                                     std::int64_t(r.numerator);
              const std::int64_t param =
                  std::int64_t(ta) +
                  (x < 0 ? (x - den / 2) : (x + den / 2)) / den;
              if (param <= 0 || std::int64_t(t_hi_v) <= param)
                continue; // lands on an endpoint: nothing to add
              local.proposals.push_back({pedge, d.tag, Index(param)});
            }
          }
        }
        // harvest pass-1 interiors: points of kept (odd-parity) faces
        // that sit on no constrained edge — walk vertices and splits
        // are ON constraints, so survivors are the refiner's interior
        // insertions. They re-enter pass 2 as plain input points.
        {
          auto rlabels = local.rf.region_labels();
          auto rp = local.rf.points();
          local.usedp.allocate(rp.size());
          local.onc.allocate(rp.size());
          std::fill(local.usedp.begin(), local.usedp.end(), char(0));
          std::fill(local.onc.begin(), local.onc.end(), char(0));
          for (Index f = 0; f < local.rf.n_faces(); ++f) {
            if ((rlabels[std::size_t(f)] & 1) == 0)
              continue;
            auto fc2 = local.rf.face(f);
            for (int e = 0; e < 3; ++e) {
              local.usedp[std::size_t(fc2[std::size_t(e)])] = 1;
              if (local.rf.constrained(f, e)) {
                local.onc[std::size_t(fc2[std::size_t(e)])] = 1;
                local.onc[std::size_t(fc2[std::size_t((e + 1) % 3)])] = 1;
              }
            }
          }
          const std::size_t before = local.steiner.size();
          // inputs (walks, splits, injected persisted steiners) are
          // never harvested — the output array is compacted, so inputs
          // are marked through the index map, not by offset
          local.is_input.allocate(rp.size());
          std::fill(local.is_input.begin(), local.is_input.end(), char(0));
          {
            const auto &fmap = local.rf.index_map().f();
            const Index n_in = local.rf.n_input_points();
            for (Index i = 0; i < n_in; ++i) {
              const Index o = fmap[std::size_t(i)];
              if (std::make_signed_t<Index>(o) >= 0 && o < Index(rp.size()))
                local.is_input[std::size_t(o)] = 1;
            }
          }
          for (std::size_t i = 0; i < rp.size(); ++i)
            if (local.usedp[i] && !local.onc[i] && !local.is_input[i])
              local.steiner.push_back(lift_to_plane(local.sc, rp[i]));
          if (local.steiner.size() != before)
            local.steiner_of.push_back(
                {Index(li_z), Index(local.steiner.size() - before)});
        }
      }
    };
    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.proposals, proposals);
      tf::core::append(local.steiner, steiner_pool);
      tf::core::append(local.steiner_of, steiner_of);
      tf::core::append(local.rf_failed, rf_failed);
      tf::core::append(local.rec_proposals, rec_proposals);
      tf::core::append(local.rec_parent_edges, rec_parent_edges);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::zip(tf::make_sequence_range(std::size_t(loops.size())), descs,
                loops, loop_holes),
        tf::none, local_t{}, task, agg);

    const std::size_t n_before = _splits.size();
    // refusal recovery: materialize discovered crossings, retry with
    // subdivided constraints; survivors are terminal, pass 2 emits
    // them plain
    n_refine_refused = Index(rf_failed.size());
    n_refine_recovered = 0;
    if (rf_failed.size() != 0) {
      const std::size_t steiner_of_before_recovery = steiner_of.size();
      auto get_point_g = [&](Index tag, const vertex_t &v) {
        return point_of(tag, v, ipts, get_mesh_point);
      };
      for (int round = 0; round < k_max_rounds && rf_failed.size() != 0;
           ++round) {
        const std::size_t n_splits_before = _splits.size();
        materialize_crossings(rec_proposals, rec_parent_edges, get_point_g);
        if (_splits.size() == n_splits_before)
          break;
        sort_splits();
        rec_proposals.clear();
        rec_parent_edges.clear();
        tf::buffer<Index> failed_now;
        std::swap(failed_now, rf_failed);
        local_t retry_local{};
        task(tf::zip(tf::make_range(failed_now),
                     tf::make_indirect_range(tf::make_range(failed_now), descs),
                     tf::make_indirect_range(tf::make_range(failed_now), loops),
                     tf::make_indirect_range(tf::make_range(failed_now),
                                             loop_holes)),
             retry_local);
        agg(retry_local, tf::none);
        n_refine_recovered +=
            Index(failed_now.size()) - Index(retry_local.rf_failed.size());
      }
      // retries append out of region order — restore the region-major
      // pool layout
      if (steiner_of.size() != steiner_of_before_recovery) {
        tf::buffer<std::array<Index, 3>> order; // (region, start, count)
        order.allocate(steiner_of.size());
        Index at = 0;
        for (std::size_t i = 0; i < steiner_of.size(); ++i) {
          order[i] = {steiner_of[i][0], at, steiner_of[i][1]};
          at += steiner_of[i][1];
        }
        std::sort(order.begin(), order.end());
        tf::buffer<tf::point<Int, 3>> pool;
        pool.allocate(steiner_pool.size());
        Index w = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
          steiner_of[i] = {order[i][0], order[i][2]};
          for (Index k = 0; k < order[i][2]; ++k)
            pool[std::size_t(w++)] =
                steiner_pool[std::size_t(order[i][1] + k)];
        }
        steiner_pool = std::move(pool);
      }
    }

    materialize_proposals(proposals, ipts, get_mesh_point);

    // harvested interiors alone still warrant pass 2 — a region whose
    // whole walk is frozen refines purely by interior insertions
    if (_splits.size() == n_before && steiner_pool.size() == 0)
      return;
    collect_original_edge_splits();

    propagate_conforming_rings(fr, n_tags, apply_to_form,
                               get_mesh_point, ipts);
    // harvested interiors materialize now, in region order; pass 2
    // receives them as plain input points (the reference's bare-CDT
    // re-injection)
    tf::buffer<Index> steiner_off;
    steiner_off.allocate(std::size_t(loops.size()) + 1);
    tf::parallel_fill(steiner_off, Index(0));
    for (const auto &so : steiner_of)
      steiner_off[std::size_t(so[0]) + 1] = so[1];
    for (std::size_t li = 0; li < std::size_t(loops.size()); ++li)
      steiner_off[li + 1] += steiner_off[li];
    const Index steiner_base = Index(extra_points.size());
    tf::core::append(steiner_pool, extra_points);

    // consumption: every carrier re-triangulates against the shared
    // splits, and the faces they reach are promoted — the same path
    // recovery uses. Live regions take their final tessellation from
    // the refiner so its interior points survive; promoted faces come
    // out of the plain path, their boundary already conformed.
    // pass 2 is a bare CDT: quality lives in pass-1's boundary splits
    // and its harvested interiors, injected above
    tf::cdt_refine_config bare = config;
    bare.min_quality = 0.f;
    bare.split_encroached = false;
    // Projected-coincidence welds are COMMON under refinement (split
    // points near seam vertices in skewed projections): the sweep
    // reports them, consolidation retires the identities globally, and
    // conform_welds substitutes survivors into every carrier — no
    // second sweep (substitution cannot create new welds).
    {
      tf::buffer<merge_t> weld_pending;
      retriangulate_all(fr, apply_to_face, get_mesh_point, ipts, sc,
                        dead_loops, &rf, &bare, &steiner_off, steiner_base,
                        &weld_pending);
      _promoted.clear();
      // promoted boundaries are final — interior quality insertion
      // only, no encroachment splits
      tf::cdt_refine_config prom_config = config;
      prom_config.split_encroached = false;
      promote(fr, _n_tags, apply_to_form, apply_to_face, get_mesh_point, ipts,
              &rf, &prom_config, &weld_pending);
      if (consolidate_merges(weld_pending))
        conform_welds(fr);
    }
    // fresh steiners join the persistent table (region-sorted);
    // weld-retired ones stay out, and previously persisted ids that a
    // later weld retired are swept here too
    {
      Index base = Index(0);
      for (const auto &so : steiner_of) {
        for (Index k = 0; k < so[1]; ++k) {
          const Index id = n_ig_created + steiner_base + base + k;
          const vertex_t v{source::created, id, {0, tf::topo_type::face}};
          if (resolve_key(Index(0), v) == id_of(Index(0), v))
            _steiners.push_back({so[0], id});
        }
        base += so[1];
      }
      if (_merge.size() != 0) {
        std::size_t w = 0;
        for (std::size_t i = 0; i < _steiners.size(); ++i) {
          const vertex_t v{source::created, _steiners[i][1],
                           {0, tf::topo_type::face}};
          if (resolve_key(Index(0), v) == id_of(Index(0), v))
            _steiners[w++] = _steiners[i];
        }
        _steiners.erase_till_end(_steiners.begin() + std::ptrdiff_t(w));
      }
      std::sort(_steiners.begin(), _steiners.end());
    }
    // promoted-face interiors persist like region steiners: weld-
    // retired ids stay out
    {
      Index base = Index(0);
      for (std::size_t k = 0; k < _promoted_interior_of.size(); ++k) {
        const auto &pd = _promoted[k];
        for (Index j = 0; j < _promoted_interior_of[k]; ++j) {
          const Index id = n_ig_created + _promoted_interior_base + base + j;
          const vertex_t v{source::created, id, {0, tf::topo_type::face}};
          if (resolve_key(Index(0), v) == id_of(Index(0), v))
            _promoted_steiners.push_back({pd.tag, pd.object, id});
        }
        base += _promoted_interior_of[k];
      }
      if (_merge.size() != 0) {
        std::size_t w = 0;
        for (std::size_t i = 0; i < _promoted_steiners.size(); ++i) {
          const vertex_t v{source::created, _promoted_steiners[i][2],
                           {0, tf::topo_type::face}};
          if (resolve_key(Index(0), v) == id_of(Index(0), v))
            _promoted_steiners[w++] = _promoted_steiners[i];
        }
        _promoted_steiners.erase_till_end(_promoted_steiners.begin() +
                                          std::ptrdiff_t(w));
      }
      std::sort(_promoted_steiners.begin(), _promoted_steiners.end());
    }
    // stack aliases re-assert after the re-splice: dead regions re-emit
    // their survivor's refined block (winding via the triple's flag)
    rev.allocate(ranges.size());
    tf::parallel_fill(rev, char(0));
    for (const auto &cp : coplanar_pairs) {
      if (dead_loops.size() != 0 && bool(dead_loops[std::size_t(cp[0])]))
        continue;
      ranges[std::size_t(cp[1])] = ranges[std::size_t(cp[0])];
      rev[std::size_t(cp[1])] = char(cp[2]);
      _aliased = true;
    }
    // record the converged state: fingerprints see the post-weld
    // steiner table and the fully materialized split table
    _region_fingerprints.allocate(std::size_t(loops.size()));
    tf::parallel_for_each(
        tf::make_sequence_range(std::size_t(loops.size())),
        [&](std::size_t li) {
          _region_fingerprints[li] = descs[Index(li)].tag == Index(-1)
                               ? std::array<Index, 2>{0, 0}
                               : region_fingerprint(fr, Index(li));
        });
    _refined_quality = config.min_quality;
    finalize(fr);
  }

  template <typename Forms, typename ApplyToFace, typename GetMeshPoint>
  auto build(const tf::face_regions<Index, Int> &fr,
             const tf::intersection_graph<Index, Int> &ig, const Forms &forms,
             const ApplyToFace &apply_to_face,
             const GetMeshPoint &get_mesh_point) -> void {
    const tf::buffer<char> no_dead;
    const tf::buffer<std::array<Index, 3>> no_pairs;
    build(fr, ig, forms, apply_to_face, get_mesh_point,
          tf::make_range(no_dead), tf::make_range(no_pairs));
  }

  /// Range convenience over @ref build_applied.
  /// @overload
  template <typename Forms, typename ApplyToFace, typename GetMeshPoint,
            typename DeadLoops, typename CoplanarPairs>
  auto build(const tf::face_regions<Index, Int> &fr,
             const tf::intersection_graph<Index, Int> &ig, const Forms &forms,
             const ApplyToFace &apply_to_face,
             const GetMeshPoint &get_mesh_point, const DeadLoops &dead_loops,
             const CoplanarPairs &coplanar_pairs) -> void {
    auto apply_form = [&forms](Index t, auto &&f) { f(forms[t]); };
    build_applied(fr, ig, apply_form, apply_to_face, get_mesh_point,
                  dead_loops, coplanar_pairs);
  }

  /// With the graph's coplanar stacks, mirroring
  /// loop_triangulations::apply_stack_aliases: `dead_loops` members are
  /// skipped outright and, once everything has settled, their `ranges`
  /// entry aliases the surviving representative's block with `rev`
  /// marking reversed winding. `coplanar_pairs` are
  /// `{survivor, dead, reversed}` triples.
  /// @overload
  template <typename ApplyToForm, typename ApplyToFace, typename GetMeshPoint,
            typename DeadLoops, typename CoplanarPairs>
  auto build_applied(const tf::face_regions<Index, Int> &fr,
                     const tf::intersection_graph<Index, Int> &ig,
                     const ApplyToForm &apply_to_form,
                     const ApplyToFace &apply_to_face,
                     const GetMeshPoint &get_mesh_point,
                     const DeadLoops &dead_loops,
                     const CoplanarPairs &coplanar_pairs) -> void {
    clear();
    auto is_dead = [&](std::size_t li) -> bool {
      return dead_loops.size() != 0 && bool(dead_loops[li]);
    };
    auto descs = fr.descriptors();
    auto loops = fr.loops();
    auto loop_holes = fr.loop_holes();
    auto ipts = ig.points();
    n_ig_created = static_cast<Index>(ipts.size());
    _n_tags = static_cast<Index>(fr.tag_offsets().size()) - 1;
    const std::size_t n_loops = std::size_t(loops.size());
    ranges.allocate(n_loops);

    struct local_t {
      scratch_t sc;
      tf::buffer<std::array<vertex_t, 3>> out;
      tf::buffer<Index> counts;
      tf::buffer<Index> failed;
    };
    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (auto &&[li_z, desc, loop, hole_ids] : range) {
        const std::size_t before = local.out.size();
        bool ok = true;
        if (desc.tag != Index(-1) && loop.size() >= 3 && !is_dead(li_z))
          ok = run_loop(fr, desc, loop, hole_ids, apply_to_face_f,
                        get_mesh_point_f, ipts, local.sc, local.out, false);
        if (!ok) {
          local.out.erase_till_end(local.out.begin() +
                                   std::ptrdiff_t(before));
          local.failed.push_back(Index(li_z));
        }
        local.counts.push_back(Index(local.out.size() - before));
      }
    };
    tf::buffer<Index> counts;
    tf::buffer<merge_t> pending;
    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.out, tris);
      tf::core::append(local.counts, counts);
      tf::core::append(local.failed, failed);
      tf::core::append(local.sc.merges, pending);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::zip(tf::make_sequence_range(n_loops), descs, loops, loop_holes),
        tf::none, local_t{}, task, agg);
    Index acc = 0;
    for (std::size_t li = 0; li < n_loops; ++li) {
      ranges[li] = {acc, acc + counts[li]};
      acc += counts[li];
    }

    consolidate_merges(pending);
    if (failed.size() || _merge.size())
      recover(fr, ig, apply_to_face, get_mesh_point, dead_loops);
    n_failed = static_cast<Index>(failed.size());
    collect_original_edge_splits();
    // Promotion-discovered welds are global facts like any other:
    // report, consolidate, and conform by substitution. With no welds
    // (the overwhelming case) this is one plain promote.
    {
      tf::buffer<merge_t> weld_pending;
      promote(fr, _n_tags, apply_to_form, apply_to_face, get_mesh_point, ipts,
              static_cast<tf::none_t *>(nullptr), nullptr, &weld_pending);
      if (consolidate_merges(weld_pending))
        conform_welds(fr);
    }

    // Stack aliasing LAST: recovery splices `ranges` and promotion
    // appends to it, so an alias written earlier would go stale (`rev`
    // spans the promoted blocks too, zero there). Cliques include
    // (dead, dead) pairs; only the direct (live survivor, dead) pair
    // carries the winding of the triangles actually aliased.
    rev.allocate(ranges.size());
    tf::parallel_fill(rev, char(0));
    for (const auto &cp : coplanar_pairs) {
      if (is_dead(std::size_t(cp[0])))
        continue;
      ranges[std::size_t(cp[1])] = ranges[std::size_t(cp[0])];
      rev[std::size_t(cp[1])] = char(cp[2]);
      _aliased = true;
    }
    finalize(fr);
  }

  /// Materialize the triangle-grain exposure (loops / descriptors /
  /// face_offsets / tag_offsets / deleted). The stream is contiguous
  /// per tag across all loop kinds; within a tag the structure's loops
  /// come in loop order, then the tag's late-discovered promoted faces.
  /// Recopied only when it must be — a promoted face or an applied
  /// alias; otherwise `tris` already is the stream and only the
  /// per-triangle descriptors materialize. Build and refine call this
  /// last.
  auto finalize(const tf::face_regions<Index, Int> &fr) -> void {
    copy_deleted(fr);
    auto descs = fr.descriptors();
    auto fr_tag_offsets = fr.tag_offsets();
    const std::size_t n_struct = descs.size();
    const std::size_t n_loops = ranges.size();
    const Index n_tags = _n_tags < 0 ? Index(0) : _n_tags;
    _tag_offsets.allocate(std::size_t(n_tags) + 1);
    _face_offsets.clear();
    _exposed_owned = _promoted.size() != 0 || _aliased;

    if (!_exposed_owned) {
      _exposed_ranges.allocate(n_loops);
      tf::parallel_copy(tf::make_range(ranges), tf::make_range(_exposed_ranges));
      const Index n_tris = static_cast<Index>(tris.size());
      _tag_offsets[0] = 0;
      for (Index t = 0; t < n_tags; ++t) {
        const std::size_t e = std::size_t(fr_tag_offsets[std::size_t(t) + 1]);
        _tag_offsets[std::size_t(t) + 1] = e < n_loops ? ranges[e][0] : n_tris;
      }
      desc_t prev{Index(-1), Index(-1)};
      for (std::size_t li = 0; li < n_loops; ++li) {
        if (ranges[li][0] == ranges[li][1])
          continue;
        const desc_t d = descs[li];
        if (d.tag != prev.tag || d.object != prev.object) {
          _face_offsets.push_back(ranges[li][0]);
          prev = d;
        }
      }
      _face_offsets.push_back(n_tris);
      _exposed_descriptors.allocate(tris.size());
      tf::parallel_for_each(tf::make_sequence_range(n_loops),
                            [&](std::size_t li) {
                              const auto r = ranges[li];
                              const desc_t d = descs[li];
                              for (Index t = r[0]; t < r[1]; ++t)
                                _exposed_descriptors[std::size_t(t)] = d;
                            });
      _exposed_triangles.clear();
      return;
    }

    tf::buffer<Index> loop_order;
    loop_order.allocate(n_loops);
    tf::buffer<std::size_t> tag_ends;
    tag_ends.allocate(std::size_t(n_tags));
    {
      std::size_t pos = 0, pk = 0;
      for (Index t = 0; t < n_tags; ++t) {
        for (Index li = fr_tag_offsets[std::size_t(t)];
             li < fr_tag_offsets[std::size_t(t) + 1]; ++li)
          loop_order[pos++] = li;
        while (pk < _promoted.size() && _promoted[pk].tag == t) {
          loop_order[pos++] = static_cast<Index>(n_struct + pk);
          ++pk;
        }
        tag_ends[std::size_t(t)] = pos;
      }
    }
    tf::buffer<Index> loop_offsets;
    loop_offsets.allocate(n_loops + 1);
    Index acc = 0;
    desc_t prev{Index(-1), Index(-1)};
    for (std::size_t pos = 0; pos < n_loops; ++pos) {
      const std::size_t li = std::size_t(loop_order[pos]);
      const desc_t d = li < n_struct ? descs[li] : _promoted[li - n_struct];
      const Index cnt = ranges[li][1] - ranges[li][0];
      loop_offsets[pos] = acc;
      if (cnt != 0 && (d.tag != prev.tag || d.object != prev.object)) {
        _face_offsets.push_back(acc);
        prev = d;
      }
      acc += cnt;
    }
    loop_offsets[n_loops] = acc;
    _face_offsets.push_back(acc);
    _tag_offsets[0] = 0;
    for (Index t = 0; t < n_tags; ++t)
      _tag_offsets[std::size_t(t) + 1] = loop_offsets[tag_ends[std::size_t(t)]];

    _exposed_ranges.allocate(n_loops);
    tf::parallel_for_each(tf::make_sequence_range(n_loops),
                          [&](std::size_t pos) {
                            _exposed_ranges[std::size_t(loop_order[pos])] = {
                                loop_offsets[pos], loop_offsets[pos + 1]};
                          });

    _exposed_triangles.allocate(std::size_t(acc));
    _exposed_descriptors.allocate(std::size_t(acc));
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t pos) {
          const std::size_t li = std::size_t(loop_order[pos]);
          const auto r = ranges[li];
          if (r[0] == r[1])
            return;
          const desc_t d = li < n_struct ? descs[li] : _promoted[li - n_struct];
          const bool rv = li < rev.size() && bool(rev[li]);
          std::size_t o = std::size_t(loop_offsets[pos]);
          for (Index t = r[0]; t < r[1]; ++t) {
            const auto &tr = tris[std::size_t(t)];
            _exposed_triangles[o] = {tr[0], tr[rv ? 2 : 1], tr[rv ? 1 : 2]};
            _exposed_descriptors[o] = d;
            ++o;
          }
        });
  }

private:
  struct hash_t {
    auto operator()(const vertex_t &v) const {
      return hash(std::array<Index, 2>{Index(v.source), v.id});
    }
    tf::array_hash<Index, 2> hash;
  };

  /// A weld observed in one face is a GLOBAL fact: the two identities
  /// name one vertex everywhere. `from` is the retired identity, `to`
  /// the survivor (the smallest identity of the welded group).
  struct merge_t {
    std::array<Index, 2> from;
    std::array<Index, 2> to_key;
    vertex_t to;
  };

  // A batch split proposal: the point it names is materialized once,
  // after fusion, so threads never touch the shared table.
  struct proposal_t {
    std::array<vertex_t, 2> edge;
    Index tag;
    Index param; // canonical dyadic, already on the parent edge
  };

  /// The unified created-point resolution: refinement extras above the
  /// watermark, intersection-graph points below, mesh points otherwise.
  template <typename Ipts, typename GetMeshPoint>
  auto point_of(Index tag, const vertex_t &v, const Ipts &ipts,
                const GetMeshPoint &get_mesh_point) const
      -> tf::point<Int, 3> {
    if (v.source == source::created) {
      if (v.id >= n_ig_created)
        return extra_points[std::size_t(v.id - n_ig_created)];
      return ipts[v.id];
    }
    return get_mesh_point(tag, v.id);
  }

  /// Fuse a batch of dyadic split proposals against the table and
  /// materialize the survivors — one point, one identity per split;
  /// returns how many entered the table.
  template <typename Ipts, typename GetMeshPoint>
  auto materialize_proposals(const tf::buffer<proposal_t> &props,
                             const Ipts &ipts,
                             const GetMeshPoint &get_mesh_point)
      -> std::size_t {
    const bool check_prior = _splits.size() != 0;
    struct rec_t {
      ekey_t edge;
      Index num;
      Index tag;
      vertex_t lo, hi;
    };
    tf::buffer<rec_t> recs;
    recs.allocate(props.size());
    tf::parallel_for_each(
        tf::enumerate(props), [&](auto pr) {
          auto &&[i, p] = pr;
          const auto k0 = id_of(p.tag, p.edge[0]);
          const auto k1 = id_of(p.tag, p.edge[1]);
          const bool fwd = k0 < k1;
          // params arrive canonical (measured from the min-identity
          // endpoint) — only the endpoint order normalizes here
          recs[std::size_t(i)] =
              rec_t{edge_key(k0, k1), p.param, p.tag,
                    fwd ? p.edge[0] : p.edge[1], fwd ? p.edge[1] : p.edge[0]};
        },
        tf::checked);
    tbb::parallel_sort(recs.begin(), recs.end(),
                       [](const rec_t &a, const rec_t &b) {
                         if (a.edge != b.edge)
                           return a.edge < b.edge;
                         return a.num < b.num;
                       });
    tf::buffer<std::size_t> uniq;
    uniq.reserve(recs.size());
    // adjacent-param fusion (the recovery path's on-edge sub-ulp ball,
    // mirrored here): a proposal within one lattice unit of the last
    // kept one, or of an existing split, is its twin — same edge means
    // the same carriers, so fusing is local and identity-clean
    Index last_num = Index(0);
    bool have_last = false;
    for (std::size_t i = 0; i < recs.size(); ++i) {
      if (i != 0 && recs[i].edge != recs[i - 1].edge)
        have_last = false;
      if (have_last && recs[i].num - last_num <= Index(1))
        continue;
      if (check_prior && has_adjacent_split(recs[i].edge, recs[i].num))
        continue;
      uniq.push_back(i);
      last_num = recs[i].num;
      have_last = true;
    }
    const Index id_base = n_ig_created + Index(extra_points.size());
    const std::size_t split_base = _splits.size();
    extra_points.reallocate(extra_points.size() + uniq.size());
    _splits.reallocate(_splits.size() + uniq.size());
    auto pt = [&](Index tag, const vertex_t &v) {
      return point_of(tag, v, ipts, get_mesh_point);
    };
    tf::parallel_for_each(
        tf::enumerate(tf::make_range(uniq)), [&](auto ur) {
          auto &&[k, ri] = ur;
          const auto &r = recs[ri];
          const auto a = pt(r.tag, r.lo);
          const auto b = pt(r.tag, r.hi);
          const T1 den = T1(1) << k_param_bits;
          const T1 nu = T1(r.num);
          tf::point<Int, 3> pp;
          for (int c = 0; c < 3; ++c)
            pp[c] = static_cast<Int>(tf::exact::div_round(
                T1(a[c]) * (den - nu) + T1(b[c]) * nu, den));
          const Index id = id_base + Index(k);
          extra_points[std::size_t(id - n_ig_created)] = pp;
          const auto v = vertex_t{source::created, id,
                                  {0, tf::topo_type::edge}};
          _splits[split_base + std::size_t(k)] =
              split_t{r.edge, r.num, id_of(r.tag, v), v};
        },
        tf::checked);
    // tail is (edge, param)-sorted and fusion excludes cross-table
    // duplicates — merge, don't re-sort
    if (uniq.size() != 0)
      std::inplace_merge(_splits.begin(),
                         _splits.begin() + std::ptrdiff_t(split_base),
                         _splits.end(),
                         [](const split_t &x, const split_t &y) {
                           if (x.edge != y.edge)
                             return x.edge < y.edge;
                           return x.param < y.param;
                         });
    return uniq.size();
  }

  /// Conforming propagation: uncut faces consuming splits on a shared
  /// original edge demand graded splits on their other simple edges —
  /// cyclic Lipschitz sizing over the subdivided boundary, rings
  /// cascade until stable. Grading runs in 256-unit param space;
  /// emitted params scale onto the lattice (<< 22, exact).
  template <typename Fr, typename ApplyToForm, typename GetMeshPoint,
            typename Ipts>
  auto propagate_conforming_rings(const Fr &fr, Index n_tags,
                                  const ApplyToForm &apply_to_form,
                                  const GetMeshPoint &get_mesh_point,
                                  const Ipts &ipts) -> void {
    auto descs = fr.descriptors();
    tf::core::std_vector<tf::buffer<char>> cut_mask(
        static_cast<std::size_t>(n_tags));
    for (Index t = 0; t < n_tags; ++t) {
      apply_to_form(t, [&](const auto &form) {
        cut_mask[std::size_t(t)].allocate(form.faces().size());
      });
      std::fill(cut_mask[std::size_t(t)].begin(), cut_mask[std::size_t(t)].end(),
                char(0));
    }
    for (Index l = 0; l < Index(descs.size()); ++l)
      if (descs[l].tag != Index(-1))
        cut_mask[std::size_t(descs[l].tag)][std::size_t(descs[l].object)] = 1;
    for (Index t = 0; t < n_tags; ++t)
      for (auto dd : fr.deleted(t))
        cut_mask[std::size_t(t)][std::size_t(dd)] = 1;

    // scan list: original edges whose split set changed this round
    tf::buffer<std::array<Index, 3>> scan; // (tag, u, v)
    for (const auto &os : _original_splits)
      scan.push_back({os.tag, os.u, os.v});
    tbb::parallel_sort(scan.begin(), scan.end());
    scan.erase_till_end(std::unique(scan.begin(), scan.end()));

    tf::buffer<std::array<Index, 2>> cand;
    tf::buffer<proposal_t> ring_props;
    struct ring_local_t {
      tf::buffer<std::array<double, 2>> fpts;
      tf::buffer<double> fh, felen;
      tf::buffer<std::size_t> fedge;
      tf::buffer<std::uint32_t> fparams;
      tf::buffer<proposal_t> props;
    };
    for (int ring = 0; ring < 4; ++ring) {
      cand.clear();
      auto emit_cand = [&](const std::array<Index, 3> &sk,
                           tf::buffer<std::array<Index, 2>> &out) {
        apply_to_form(sk[0], [&](const auto &form) {
          for (auto f_id : form.face_membership()[sk[1]]) {
            if (cut_mask[std::size_t(sk[0])][std::size_t(f_id)])
              continue;
            auto face = form.faces()[f_id];
            const Index n = Index(face.size());
            for (Index e = 0; e < n; ++e) {
              const Index a = Index(face[std::size_t(e)]);
              const Index b = Index(face[std::size_t((e + 1) % n)]);
              if ((a == sk[1] && b == sk[2]) || (a == sk[2] && b == sk[1])) {
                out.push_back({sk[0], Index(f_id)});
                break;
              }
            }
          }
        });
      };
      if (scan.size() < 512) {
        for (const auto &sk : scan)
          emit_cand(sk, cand);
      } else {
        struct disc_local_t {
          tf::buffer<std::array<Index, 2>> found;
        };
        tf::blocked_reduce_sequenced_aggregate(
            tf::make_range(scan), tf::none, disc_local_t{},
            [&](auto &&range2, disc_local_t &local) {
              for (const auto &sk : range2)
                emit_cand(sk, local.found);
            },
            [&](const disc_local_t &local, const tf::none_t &) {
              tf::core::append(local.found, cand);
            });
      }
      tbb::parallel_sort(cand.begin(), cand.end());
      cand.erase_till_end(std::unique(cand.begin(), cand.end()));
      if (cand.size() == 0)
        break;
      ring_props.clear();
      auto ring_task = [&](auto &&range, ring_local_t &local) {
        auto get_mesh_point_f = get_mesh_point;
        for (const auto &c : range) {
          const Index tag = c[0], f_id = c[1];
          apply_to_form(tag, [&](const auto &form) {
            auto face = form.faces()[f_id];
            const std::size_t n = face.size();
            const auto p0 = get_mesh_point_f(tag, Index(face[0]));
            const auto p1 = get_mesh_point_f(tag, Index(face[1]));
            const auto p2 = get_mesh_point_f(tag, Index(face[2]));
            const auto axes = tf::exact::projection_axes(p0, p1, p2);
            local.fpts.clear();
            local.fedge.clear();
            local.fparams.clear();
            for (std::size_t e = 0; e < n; ++e) {
              const Index u = Index(face[e]);
              const Index v = Index(face[(e + 1) % n]);
              const auto pu = get_mesh_point_f(tag, u);
              const auto pv = get_mesh_point_f(tag, v);
              const std::array<double, 2> xu{double(pu[axes.first]),
                                             double(pu[axes.second])};
              const std::array<double, 2> xv{double(pv[axes.first]),
                                             double(pv[axes.second])};
              local.fpts.push_back(xu);
              local.fedge.push_back(e);
              local.fparams.push_back(0);
              const auto vu = vertex_t{source::original, u,
                                       {0, tf::topo_type::vertex}};
              const auto vv = vertex_t{source::original, v,
                                       {0, tf::topo_type::vertex}};
              const auto ku = id_of(tag, vu);
              const auto kv = id_of(tag, vv);
              const bool fwd = ku < kv;
              auto [s0, s1] = splits_of(edge_key(ku, kv));
              for (Index k = 0; k < s1 - s0; ++k) {
                const Index idx = fwd ? s0 + k : s1 - 1 - k;
                // round to the sizing grid; a split that quantizes
                // onto a corner (or its bucket neighbour) carries no
                // sizing feature — keeping it would fabricate a
                // zero-length cycle edge and grade toward nothing
                std::uint32_t p8 = std::uint32_t(
                    (_splits[std::size_t(idx)].param +
                     (Index(1) << (k_param_bits - 9))) >>
                    (k_param_bits - 8));
                if (!fwd)
                  p8 = 256u - p8;
                if (p8 == 0u || 256u <= p8)
                  continue;
                if (local.fedge.back() == e && local.fparams.back() == p8)
                  continue;
                const double t8 = double(p8) / 256.0;
                local.fpts.push_back({xu[0] + t8 * (xv[0] - xu[0]),
                                      xu[1] + t8 * (xv[1] - xu[1])});
                local.fedge.push_back(e);
                local.fparams.push_back(p8);
              }
            }
            const std::size_t m = local.fpts.size();
            detail::cycle_sizing(local.fpts, false, local.fh, local.felen);
            auto no_lfs = [](const std::array<double, 2> &) {
              return 1e300;
            };
            auto link_rows = form.manifold_edge_link()[f_id];
            for (std::size_t i = 0; i < m; ++i) {
              const std::size_t in = (i + 1) % m;
              const std::size_t e = local.fedge[i];
              if (!link_rows[e].is_simple())
                continue;
              const std::uint32_t pa = local.fparams[i];
              const std::uint32_t pb =
                  (in != 0 && local.fedge[in] == e) ? local.fparams[in]
                                                    : 256u;
              const Index u = Index(face[e]);
              const Index v = Index(face[(e + 1) % n]);
              const auto vu = vertex_t{source::original, u,
                                       {0, tf::topo_type::vertex}};
              const auto vv = vertex_t{source::original, v,
                                       {0, tf::topo_type::vertex}};
              const bool fwd = id_of(tag, vu) < id_of(tag, vv);
              detail::dyadic_split(
                  pa, pb, local.fpts[i], local.fpts[in], local.fh[i],
                  local.fh[in],
                  [&](std::uint32_t p) {
                    const std::uint32_t pc = fwd ? p : 256u - p;
                    local.props.push_back(
                        {{vu, vv}, tag,
                         Index(pc) << (k_param_bits - 8)});
                  },
                  no_lfs);
            }
          });
        }
      };
      auto ring_agg = [&](const ring_local_t &local, const tf::none_t &) {
        tf::core::append(local.props, ring_props);
      };
      tf::blocked_reduce_sequenced_aggregate(tf::make_range(cand), tf::none,
                                             ring_local_t{}, ring_task,
                                             ring_agg);
      if (ring_props.size() == 0)
        break;
      scan.clear();
      for (const auto &rp : ring_props)
        scan.push_back({rp.tag, rp.edge[0].id, rp.edge[1].id});
      tbb::parallel_sort(scan.begin(), scan.end());
      scan.erase_till_end(std::unique(scan.begin(), scan.end()));
      if (materialize_proposals(ring_props, ipts, get_mesh_point) == 0)
        break;
    }
    collect_original_edge_splits();
  }

  // A discovery proposal carries everything materialization needs,
  // resolved where it must be: a T-junction names a loop-local input
  // vertex, unreachable after the loop's scratch is gone. Its parent
  // edges live in a parallel pool, `n_parents` per proposal.
  struct rec_proposal_t {
    Index tag;
    Index n_parents;
    char t_junction;
    vertex_t v; // valid when `t_junction`
  };

  /// Crossing discovery on an assembled scratch: the discovery CDT is
  /// allowed to split constraints and reports each crossing/T-junction
  /// as records only — materialization stays serial at the caller.
  template <typename Sc>
  auto discover_crossings(Sc &sc, Index tag,
                          tf::buffer<rec_proposal_t> &out_props,
                          tf::buffer<std::array<vertex_t, 2>> &out_parents)
      -> void {
    auto edges = tf::make_edges(tf::make_blocked_range<2>(
        tf::make_range(sc.cons.data(), sc.cons.size())));
    sc.tri.clear();
    const bool disco =
        sc.tri.build(tf::make_points(sc.pts), edges,
                     tf::make_constant_range(true, edges.size()),
                     /*split_constraints=*/true);
    if (!disco)
      return;
    sc.tri.for_each_constraint_crossing([&](Index pt, const auto &parents) {
      // pick two parents on DISTINCT physical edges: a doubled
      // edge (slit) contributes both copies, and the closest
      // approach of a segment with itself is a bogus midpoint
      const Index a = parents[0];
      Index b = Index(-1);
      const auto &ea = sc.cons_verts[std::size_t(a)];
      const auto ka = edge_key(id_of(tag, ea[0]), id_of(tag, ea[1]));
      for (std::size_t pi = 1; pi < parents.size(); ++pi) {
        const auto &ec = sc.cons_verts[std::size_t(parents[pi])];
        if (edge_key(id_of(tag, ec[0]), id_of(tag, ec[1])) != ka) {
          b = parents[pi];
          break;
        }
      }
      if (b == Index(-1) && parents.size() > 1)
        return; // only copies of one edge — nothing to resolve
      rec_proposal_t pr{tag, Index(parents.size()), char(0), vertex_t{}};
      if (b == Index(-1)) {
        // T-junction: the point is an existing input vertex
        const auto &kf = sc.tri.index_map().kept_ids();
        const Index n_input = Index(sc.tri.index_map().f().size());
        if (kf[std::size_t(pt)] == n_input)
          return;
        pr.t_junction = char(1);
        pr.v = sc.ihm.kept_ids()[std::size_t(kf[std::size_t(pt)])];
      }
      for (std::size_t pi = 0; pi < parents.size(); ++pi)
        out_parents.push_back(sc.cons_verts[std::size_t(parents[pi])]);
      out_props.push_back(pr);
    });
  }

  /// Serial materialization of crossing records: one point, one
  /// identity (sub-ulp fusion, originals-first), registered on every
  /// distinct parent edge. The ordered stream hands out extra-point
  /// ids deterministically.
  template <typename GetPointG>
  auto materialize_crossings(const tf::buffer<rec_proposal_t> &proposals,
                             const tf::buffer<std::array<vertex_t, 2>> &parent_edges,
                             const GetPointG &get_point_g) -> void {
    std::size_t cursor = 0;
    for (const auto &pr : proposals) {
      const std::size_t base = cursor;
      cursor += std::size_t(pr.n_parents);
      vertex_t v = pr.v;
      // a T-junction names an existing vertex; a crossing does too
      // when sub-ulp fusion picks a candidate over a fresh twin
      bool named = bool(pr.t_junction);
      if (!pr.t_junction) {
        const auto &ea = parent_edges[base];
        const auto ka = edge_key(id_of(pr.tag, ea[0]), id_of(pr.tag, ea[1]));
        std::size_t bi = base;
        for (std::size_t pi = base + 1; pi < cursor; ++pi) {
          const auto &ec = parent_edges[pi];
          if (edge_key(id_of(pr.tag, ec[0]), id_of(pr.tag, ec[1])) != ka) {
            bi = pi;
            break;
          }
        }
        const auto &eb = parent_edges[bi];
        auto p = tf::exact::edge_edge_closest_point(
            get_point_g(pr.tag, ea[0]), get_point_g(pr.tag, ea[1]),
            get_point_g(pr.tag, eb[0]), get_point_g(pr.tag, eb[1]));
        // sub-ulp fusion (the ig rule): a crossing within a hair of
        // an existing vertex splits both parents AT that vertex —
        // birthing a twin would only re-cross after rounding
        auto sub_ulp = [&](const vertex_t &cand) {
          auto q = get_point_g(pr.tag, cand);
          T2 dist(0);
          for (int c = 0; c < 3; ++c) {
            T1 t = T1(q[c]) - T1(p[c]);
            dist += T2(t) * T2(t);
          }
          return !(T2(8) < dist);
        };
        // among in-range candidates the smallest identity wins — the
        // same originals-first rule as the weld survivor, and
        // independent of the parent-edge order
        const std::array<vertex_t, 4> cands{ea[0], ea[1], eb[0], eb[1]};
        bool fused = false;
        for (const auto &cand : cands)
          if (sub_ulp(cand) &&
              (!fused || id_of(pr.tag, cand) < id_of(pr.tag, v))) {
            v = cand;
            fused = true;
          }
        if (!fused) {
          v = vertex_t{source::created,
                       n_ig_created + Index(extra_points.size()),
                       {0, tf::topo_type::edge}};
          extra_points.push_back(p);
        }
        named = fused;
      }
      // every distinct physical edge through this point gets the
      // split: a concurrency of three or more leaves the unsplit
      // ones crossing forever
      for (std::size_t pi = base; pi < cursor; ++pi)
        register_split(pr.tag, parent_edges[pi], v, get_point_g, named);
    }
  }

  struct scratch_t {
    tf::constrained_delaunay_triangulator<Index, Int, Int> tri;
    tf::index_hash_map<vertex_t, Index, hash_t> ihm;
    tf::points_buffer<Int, 2> pts;
    tf::buffer<Index> cons;
    tf::buffer<std::array<vertex_t, 2>> cons_verts;
    tf::buffer<Index> cons_slot;
    tf::buffer<char> cons_sub;
    // for sub-edges: the parent walk edge and the canonical dyadic
    // params at the sub-edge's endpoints — refine proposals re-key
    // onto the parent so every carrier consumes them
    tf::buffer<std::array<vertex_t, 2>> cons_parent;
    tf::buffer<std::array<Index, 2>> cons_t;
    tf::buffer<std::array<Index, 3>> split_seen; // (identity key, local id)
    tf::buffer<char> pos_set;
    tf::buffer<Index> rep;
    tf::buffer<merge_t> merges;
    // this face's projection and supporting plane, for lifting points
    // the refiner inserts back into 3D
    int ax0 = 0, ax1 = 1;
    bool flip = false;
    std::array<T2, 3> plane_n{};
    T2 plane_d{};
    Int plane_fallback{};
    // face the projection/plane above belong to — loops of one face
    // are contiguous, so the memo turns a per-region recompute into a
    // per-face one
    Index memo_tag = -1;
    Index memo_object = -1;
    Index face_size = 3;
  };

  static constexpr int k_param_bits = 30;
  struct split_t {
    ekey_t edge;
    // dyadic parameter along the edge from its min-identity endpoint:
    // numerator over 2^k_param_bits. Consumption is multiply + shift —
    // the rational (int256-division) form was ~half of refine (doc 18).
    Index param;
    std::array<Index, 2> key; // identity of `vertex`
    vertex_t vertex;
    // names a pre-existing identity (recovery sub-ulp fusion): placed
    // at its projected position so it coincides with its walk copy
    char named = 0;
  };

  /// Position of a split vertex ON the edge that carries it, derived
  /// from its parameter rather than carried as geometry. A convex
  /// combination of the endpoints, so it lands exactly on the walk edge
  /// in the carrier's own projection — no fold — and every carrier of
  /// the edge derives the same lattice point (doc §18).
  auto split_point_at(const tf::point<Int, 3> &a, const tf::point<Int, 3> &b,
                      Index param, int c0, int c1) const
      -> tf::point<Int, 2> {
    const T2 w = T2((T1(1) << k_param_bits) - T1(param));
    const T2 q = T2(T1(param));
    auto round_shift = [](T2 x) -> Int {
      const T2 half = T2(T1(1) << (k_param_bits - 1));
      return x < T2(0) ? -static_cast<Int>((-x + half) >> k_param_bits)
                       : static_cast<Int>((x + half) >> k_param_bits);
    };
    return tf::point<Int, 2>{round_shift(T2(a[c0]) * w + T2(b[c0]) * q),
                             round_shift(T2(a[c1]) * w + T2(b[c1]) * q)};
  }

  /// The identity pair: `{tag, original id}`, or `{n_tags, created id}`
  /// for created points, which are tag-independent. The tag is
  /// load-bearing — the split and weld tables span every form, and
  /// original ids repeat across them. Created points sit one past the
  /// last tag so plain order puts every original first: weld survivors
  /// are the smallest identity, and an original must always win — uncut
  /// faces share it, never see this triangulation, and cannot learn a
  /// new identity.
  auto id_of(Index tag, const vertex_t &v) const -> std::array<Index, 2> {
    return v.source == source::created ? std::array<Index, 2>{_n_tags, v.id}
                                       : std::array<Index, 2>{tag, v.id};
  }
  auto is_created(const std::array<Index, 2> &k) const -> bool {
    return k[0] == _n_tags;
  }
  static auto edge_key(const std::array<Index, 2> &ka,
                       const std::array<Index, 2> &kb) -> ekey_t {
    return ka < kb ? ekey_t{ka, kb} : ekey_t{kb, ka};
  }

  /// Triangulate one region. With `use_splits`, every walk edge is
  /// augmented by the recovery splits registered on its physical edge.
  /// Fills `sc.cons_verts` (per-constraint endpoints) for discovery.
  template <typename Desc, typename Loop, typename HoleIds,
            typename ApplyToFace, typename GetMeshPoint, typename Ipts>
  auto run_loop(const tf::face_regions<Index, Int> &fr, const Desc &desc,
                const Loop &loop, const HoleIds &hole_ids,
                const ApplyToFace &apply_to_face,
                const GetMeshPoint &get_mesh_point, const Ipts &ipts,
                scratch_t &sc, tf::buffer<std::array<vertex_t, 3>> &out,
                bool use_splits, bool assemble_only = false) -> bool {
    const Index tag = desc.tag;
    auto get_point = [&](const vertex_t &v) -> tf::point<Int, 3> {
      if (v.source == source::created) {
        if (v.id >= n_ig_created)
          return extra_points[std::size_t(v.id - n_ig_created)];
        return ipts[v.id];
      }
      return get_mesh_point(tag, v.id);
    };
    bool ok = true;
    auto holes = fr.holes();
    if (sc.memo_tag != tag || sc.memo_object != desc.object) {
      apply_to_face(tag, desc.object, [&](const auto &face) {
        sc.face_size = static_cast<Index>(face.size());
        auto a = get_point(vertex_t{source::original, Index(face[0]),
                                    {0, tf::topo_type::vertex}});
        auto b = get_point(vertex_t{source::original, Index(face[1]),
                                    {1, tf::topo_type::vertex}});
        auto c = get_point(vertex_t{source::original, Index(face[2]),
                                    {2, tf::topo_type::vertex}});
        auto axes = tf::exact::projection_axes(a, b, c);
        sc.ax0 = int(axes.first);
        sc.ax1 = int(axes.second);
        T1 u[3], v3[3];
        for (int i = 0; i < 3; ++i) {
          u[i] = T1(b[i]) - T1(a[i]);
          v3[i] = T1(c[i]) - T1(a[i]);
        }
        sc.plane_n = {T2(u[1]) * T2(v3[2]) - T2(u[2]) * T2(v3[1]),
                      T2(u[2]) * T2(v3[0]) - T2(u[0]) * T2(v3[2]),
                      T2(u[0]) * T2(v3[1]) - T2(u[1]) * T2(v3[0])};
        sc.plane_d = sc.plane_n[0] * T2(a[0]) + sc.plane_n[1] * T2(a[1]) +
                     sc.plane_n[2] * T2(a[2]);
        sc.plane_fallback = a[3 - sc.ax0 - sc.ax1];
      });
      sc.memo_tag = tag;
      sc.memo_object = desc.object;
    }
    {
      const std::pair<int, int> axes{sc.ax0, sc.ax1};
      auto projected = [&](const vertex_t &v) -> tf::point<Int, 2> {
        auto pt = get_point(v);
        return {pt[axes.first], pt[axes.second]};
      };

      // positional loops never touch the map; sweeping the bucket
      // array per loop is pure waste, so clear only after a user
      if (sc.ihm.f().size())
        sc.ihm.f().clear();
      sc.ihm.kept_ids().clear();
      sc.pts.clear();
      sc.pos_set.clear();
      // The identity map dedups: walk positions resolving to one
      // identity must share a CDT point, and the smaller point set
      // saves more in incircle/orient2d than the map costs (measured:
      // positional-for-everything is identical but slower).
      // recovery/refine runs (`use_splits`) stay mapped: their discovery
      // reads kept_ids for T-junction identities, which the positional
      // path no longer fills
      const bool positional =
          !use_splits && hole_ids.size() == 0 && _merge.size() == 0;
      auto append = [&](const vertex_t &v) -> Index {
        auto id = static_cast<Index>(sc.ihm.kept_ids().size());
        sc.ihm.kept_ids().push_back(v);
        sc.pts.push_back(projected(v));
        sc.pos_set.push_back(0);
        return id;
      };
      auto local_id = [&](const vertex_t &vin) -> Index {
        vertex_t v = resolve(tag, vin);
        // a substituted survivor carries another face's sub_id — claim
        // nothing rather than the wrong original edge (the triangle
        // consumers resolve face edges through sub_ids)
        if (v.source != vin.source || v.id != vin.id)
          v.sub_id = {0, tf::topo_type::face};
        // map-free: same-identity duplicates carry identical
        // coordinates, so the CDT's coordinate weld unifies them
        if (assemble_only)
          return append(v);
        auto it = sc.ihm.f().find(v);
        if (it != sc.ihm.f().end())
          return it->second;
        auto id = append(v);
        sc.ihm.f()[v] = id;
        return id;
      };
      sc.cons.clear();
      sc.cons_verts.clear();
      sc.cons_slot.clear();
      sc.cons_sub.clear();
      sc.cons_parent.clear();
      sc.cons_t.clear();
      sc.split_seen.clear();
      T2 area2(0);
      Index walk_base = 0;
      Index cur_slot = 0;
      bool in_split = false;
      std::array<vertex_t, 2> cur_parent{};
      std::array<Index, 2> cur_t{};
      auto add_edge = [&](const vertex_t &va, const vertex_t &vb) {
        const Index a = local_id(va);
        const Index b = local_id(vb);
        if (a == b)
          return;
        // Distinct identities can share this face's projected position
        // (a split that collapses onto an endpoint here but not in its
        // own face). Emitting the segment would be a degenerate
        // constraint; the points stay and weld, the canonical
        // representative rule keeps identity consistent.
        auto pa2 = sc.pts[std::size_t(a)];
        auto pb2 = sc.pts[std::size_t(b)];
        if (pa2[0] == pb2[0] && pa2[1] == pb2[1])
          return;
        sc.cons.push_back(a);
        sc.cons.push_back(b);
        sc.cons_verts.push_back({va, vb});
        sc.cons_slot.push_back(cur_slot);
        sc.cons_sub.push_back(char(in_split));
        sc.cons_parent.push_back(in_split ? cur_parent
                                          : std::array<vertex_t, 2>{va, vb});
        sc.cons_t.push_back(in_split ? cur_t
                                     : std::array<Index, 2>{Index(0), Index(0)});
      };
      auto add_walk = [&](const auto &walk, bool with_area) {
        const auto m = walk.size();
        for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
          const auto &va = walk[j];
          const auto &vb = walk[i];
          cur_slot = walk_base + Index(j);
          if (with_area) {
            auto p = projected(va);
            auto q = projected(vb);
            area2 += T2(T1(p[0]) * T1(q[1]) - T1(q[0]) * T1(p[1]));
          }
          if (!use_splits) {
            add_edge(va, vb);
            continue;
          }
          auto key = edge_key(id_of(tag, va), id_of(tag, vb));
          auto [s0, s1] = splits_of(key);
          if (s0 == s1) {
            add_edge(va, vb);
            continue;
          }
          const bool fwd = id_of(tag, va) < id_of(tag, vb);
          const auto pa = get_point(va);
          const auto pb = get_point(vb);
          // the parameter is measured from the min-identity endpoint
          const auto &plo = fwd ? pa : pb;
          const auto &phi = fwd ? pb : pa;
          // a split lies on this walk edge, so its per-carrier copy
          // inherits the original face edge the walk edge lies on —
          // the triangle-grain consumers resolve face edges through
          // sub_ids, and the shared split record cannot carry a
          // face-local one
          tf::topo_id<short> split_sub{0, tf::topo_type::face};
          if (auto fe = tf::cut::resolve_face_edge(va.sub_id, vb.sub_id,
                                                   std::size_t(sc.face_size)))
            split_sub = {*fe, tf::topo_type::edge};
          vertex_t prev = va;
          const Index t_hi_v = Index(1) << k_param_bits;
          cur_parent = {va, vb};
          Index t_prev = fwd ? Index(0) : t_hi_v;
          for (Index k = 0; k < s1 - s0; ++k) {
            const auto &sp = _splits[std::size_t(fwd ? s0 + k : s1 - 1 - k)];
            auto sp_vertex = sp.vertex;
            sp_vertex.sub_id = split_sub;
            Index id = Index(-1);
            const auto sp_key = id_of(tag, sp_vertex);
            if (assemble_only) {
              for (const auto &e : sc.split_seen)
                if (e[0] == sp_key[0] && e[1] == sp_key[1]) {
                  id = e[2];
                  break;
                }
              if (id == Index(-1)) {
                id = local_id(sp_vertex);
                sc.split_seen.push_back({sp_key[0], sp_key[1], id});
              } else {
                // discovering face: the crossing splits two of its own
                // edges — only the canonical point satisfies both
                sc.pts[std::size_t(id)] = projected(sp_vertex);
                sc.pos_set[std::size_t(id)] = 1;
              }
            } else {
              id = local_id(sp_vertex);
            }
            // First carrier takes the on-edge parametric point; a
            // repeat means this face is the discovering face (the
            // crossing splits two of its own edges) — there only the
            // canonical projected point satisfies both (doc §14).
            if ((assemble_only && bool(sp.named)) ||
                sc.pos_set[std::size_t(id)]) {
              // a fused split IS an existing vertex: in the map-free
              // pass its projected position makes the walk copy
              // coincide (the mapped path shares one slot and keeps
              // the on-edge parametric point)
              sc.pts[std::size_t(id)] = projected(sp_vertex);
              sc.pos_set[std::size_t(id)] = 1;
            } else {
              sc.pts[std::size_t(id)] = split_point_at(
                  plo, phi, sp.param, axes.first, axes.second);
              sc.pos_set[std::size_t(id)] = 1;
            }
            cur_t = {t_prev, sp.param};
            in_split = true;
            add_edge(prev, sp_vertex);
            in_split = false;
            t_prev = sp.param;
            prev = sp_vertex;
          }
          cur_t = {t_prev, fwd ? t_hi_v : Index(0)};
          in_split = true;
          add_edge(prev, vb);
          in_split = false;
        }
        walk_base += Index(m);
      };
      if (positional) {
        // the walk IS the id space: no ids, no kept copies — emission
        // reads the loop directly, exactly like the reference path
        const auto m = loop.size();
        for (std::size_t i = 0; i < m; ++i)
          sc.pts.push_back(projected(loop[i]));
        for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
          auto p = sc.pts[j];
          auto q = sc.pts[i];
          area2 += T2(T1(p[0]) * T1(q[1]) - T1(q[0]) * T1(p[1]));
          if (p[0] == q[0] && p[1] == q[1])
            continue;
          sc.cons.push_back(Index(j));
          sc.cons.push_back(Index(i));
          sc.cons_verts.push_back({loop[j], loop[i]});
          sc.cons_slot.push_back(Index(j));
          sc.cons_sub.push_back(char(0));
          sc.cons_parent.push_back({loop[j], loop[i]});
          sc.cons_t.push_back({Index(0), Index(0)});
        }
      } else {
        add_walk(loop, true);
        for (auto h : hole_ids)
          add_walk(holes[h], false);
      }
      const bool triangle_region =
          !use_splits && hole_ids.size() == 0 && loop.size() == 3;
      std::array<Index, 3> triangle_inputs{};
      if (triangle_region)
        for (std::size_t c = 0; c < 3; ++c)
          triangle_inputs[c] =
              positional ? Index(c) : local_id(loop[c]);

      // Refined passes re-triangulate through the refiner immediately —
      // its index map carries the weld semantics, so the internal CDT
      // (and its emit) is pure duplicate work there.
      sc.flip = T2(0) < area2;
      if (assemble_only)
        return ok;

      const auto &kept_ids = sc.ihm.kept_ids();
      const std::size_t n_in = positional ? loop.size() : kept_ids.size();
      auto vat = [&](std::size_t i) -> const vertex_t & {
        return positional ? loop[i] : kept_ids[i];
      };
      auto edges = tf::make_edges(tf::make_blocked_range<2>(
          tf::make_range(sc.cons.data(), sc.cons.size())));
      sc.tri.clear();
      const bool built =
          sc.tri.build(tf::make_points(sc.pts), edges,
                       tf::make_constant_range(true, edges.size()),
                       /*split_constraints=*/false);

      // Welds are rare: with none, f() is a bijection and identity is
      // unambiguous. Otherwise the canonical representative per welded
      // vertex is the smallest identity — every caller picks the same
      // survivor regardless of input order.
      const auto &fmap = sc.tri.index_map().f();
      const auto &kf = sc.tri.index_map().kept_ids();
      const Index n_final = Index(sc.tri.points().size());
      const bool welded = std::size_t(n_final) != n_in;
      sc.rep.clear();
      if (welded) {
        sc.rep.allocate(std::size_t(n_final));
        std::fill(sc.rep.begin(), sc.rep.end(), Index(-1));
        for (Index i = 0; i < Index(n_in); ++i) {
          const Index fin = fmap[std::size_t(i)];
          if (fin < 0 || n_final <= fin) {
            ok = false;
            return ok;
          }
          auto &r = sc.rep[std::size_t(fin)];
          if (r == Index(-1) || id_of(tag, vat(std::size_t(i))) <
                                    id_of(tag, vat(std::size_t(r))))
            r = i;
        }
        // report the weld globally: every other face must retire the
        // same identities in favour of the same survivor
        for (Index i = 0; i < Index(n_in); ++i) {
          const Index r = sc.rep[std::size_t(fmap[std::size_t(i)])];
          if (r != Index(-1) && r != i)
            sc.merges.push_back({id_of(tag, vat(std::size_t(i))),
                                 id_of(tag, vat(std::size_t(r))),
                                 vat(std::size_t(r))});
        }
      }

      // A triangular region already is its topology. Coordinate welds
      // may collapse two corners to one canonical identity; only that
      // topological collapse authorizes dropping it. Area never does.
      if (triangle_region) {
        std::array<Index, 3> final{};
        for (std::size_t c = 0; c < 3; ++c) {
          final[c] = fmap[std::size_t(triangle_inputs[c])];
          if (final[c] < 0 || n_final <= final[c]) {
            ok = false;
            return ok;
          }
        }
        if (final[0] == final[1] || final[1] == final[2] ||
            final[0] == final[2])
          return ok;
        std::array<vertex_t, 3> triangle;
        for (std::size_t c = 0; c < 3; ++c) {
          const Index r = welded ? sc.rep[std::size_t(final[c])]
                                 : kf[std::size_t(final[c])];
          if (r == Index(-1)) {
            ok = false;
            return ok;
          }
          triangle[c] = vat(std::size_t(r));
        }
        out.push_back(triangle);
        return ok;
      }

      if (!built) {
        ok = false;
        return ok;
      }
      auto cdt_faces = sc.tri.make_faces();
      auto labels = sc.tri.region_labels();
      const bool flip = T2(0) < area2;
      sc.flip = flip;
      std::size_t fi = 0;
      for (auto ftri : cdt_faces) {
        if ((labels[fi++] & 1) == 0)
          continue;
        const Index r0 = welded ? sc.rep[std::size_t(ftri[0])] : kf[ftri[0]];
        const Index r1 = welded ? sc.rep[std::size_t(ftri[1])] : kf[ftri[1]];
        const Index r2 = welded ? sc.rep[std::size_t(ftri[2])] : kf[ftri[2]];
        if (r0 == Index(-1) || r1 == Index(-1) || r2 == Index(-1)) {
          ok = false;
          return ok;
        }
        if (flip)
          out.push_back({vat(std::size_t(r0)), vat(std::size_t(r2)),
                         vat(std::size_t(r1))});
        else
          out.push_back({vat(std::size_t(r0)), vat(std::size_t(r1)),
                         vat(std::size_t(r2))});
      }
    }
    return ok;
  }

public:
  /// Globally canonical tagged identity after welds.
  auto resolve_key(Index tag, const vertex_t &v) const
      -> std::array<Index, 2> {
    auto k = id_of(tag, v);
    if (_merge.size() == 0)
      return k;
    auto it = std::lower_bound(
        _merge.begin(), _merge.end(), k,
        [](const merge_t &m, const std::array<Index, 2> &x) {
          return m.from < x;
        });
    return (it != _merge.end() && it->from == k) ? it->to_key : k;
  }

  /// Carrier-local vertex after global welds. An original vertex is
  /// meaningful only together with its owning tag, which `vertex_t`
  /// intentionally does not store. When the global root belongs to
  /// another form, use the smallest class member representable by this
  /// carrier so coincident local members still collapse to one vertex.
  auto resolve(Index tag, const vertex_t &v) const -> vertex_t {
    if (_merge.size() == 0)
      return v;
    auto k = id_of(tag, v);
    auto it = std::lower_bound(
        _merge.begin(), _merge.end(), k,
        [](const merge_t &m, const std::array<Index, 2> &x) {
          return m.from < x;
        });
    if (it == _merge.end() || it->from != k)
      return v;
    if (is_created(it->to_key) || it->to_key[0] == tag)
      return it->to;

    auto local = k;
    for (const auto &merge : _merge) {
      if (merge.to_key != it->to_key)
        continue;
      if ((is_created(merge.from) || merge.from[0] == tag) &&
          merge.from < local)
        local = merge.from;
    }
    if (local == k)
      return v;
    return vertex_t{is_created(local) ? source::created : source::original,
                    local[1], {0, tf::topo_type::face}};
  }

  /// Walk the ordered splits along the physical edge (a, b), oriented
  /// from `a` — seam consumers traverse the same sub-chain every
  /// carrier emitted.
  template <typename Emit>
  auto for_each_edge_split(Index tag, const vertex_t &a, const vertex_t &b,
                           const Emit &emit) const -> void {
    const auto ka = id_of(tag, a);
    const auto kb = id_of(tag, b);
    const auto [s0, s1] = splits_of(edge_key(ka, kb));
    const bool fwd = ka < kb;
    for (Index t = Index(0); t < s1 - s0; ++t)
      emit(_splits[std::size_t(fwd ? s0 + t : s1 - 1 - t)].vertex);
  }

private:
  /// Fold newly reported welds into the table: dedup, keep the smallest
  /// survivor per retired identity, then close transitively.
  /// Conform every carrier to the consolidated welds by SUBSTITUTION
  /// (Žiga's design, the ig merge-remap precedent): resolve every
  /// emitted triangle vertex through the closed merge table and drop
  /// triangles that collapse into edges — no re-triangulation. The
  /// survivor's coordinates ride in through its identity; identity
  /// topology (edge pairing across carriers) is exact by construction.
  template <typename Int2 = Int>
  auto conform_welds(const tf::face_regions<Index, Int2> &fr) -> void {
    if (_merge.size() == 0)
      return;
    auto descs = fr.descriptors();
    const std::size_t n_struct = descs.size();
    const std::size_t n_loops = ranges.size();
    tf::buffer<Index> counts;
    counts.allocate(n_loops);
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t li) {
          const Index tag = li < n_struct
                                ? descs[li].tag
                                : _promoted[li - n_struct].tag;
          const auto r = ranges[li];
          Index kept = 0;
          for (Index t = r[0]; t < r[1]; ++t) {
            auto &tr = tris[std::size_t(t)];
            for (int c = 0; c < 3; ++c) {
              const auto &v = tr[std::size_t(c)];
              auto rv = resolve(tag, v);
              // the survivor carries the reporting face's sub_id —
              // claim nothing rather than the wrong original edge
              if (rv.source != v.source || rv.id != v.id)
                rv.sub_id = {0, tf::topo_type::face};
              tr[std::size_t(c)] = rv;
            }
            const auto k0 = resolve_key(tag, tr[0]);
            const auto k1 = resolve_key(tag, tr[1]);
            const auto k2 = resolve_key(tag, tr[2]);
            if (!(k0 == k1 || k1 == k2 || k0 == k2))
              ++kept;
          }
          counts[li] = kept;
        });
    tf::buffer<std::array<Index, 2>> new_ranges;
    new_ranges.allocate(n_loops);
    Index acc = 0;
    for (std::size_t li = 0; li < n_loops; ++li) {
      new_ranges[li] = {acc, acc + counts[li]};
      acc += counts[li];
    }
    tf::buffer<std::array<vertex_t, 3>> out;
    out.allocate(std::size_t(acc));
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t li) {
          const Index tag = li < n_struct
                                ? descs[li].tag
                                : _promoted[li - n_struct].tag;
          const auto r = ranges[li];
          Index w = new_ranges[li][0];
          for (Index t = r[0]; t < r[1]; ++t) {
            const auto &tr = tris[std::size_t(t)];
            const auto k0 = resolve_key(tag, tr[0]);
            const auto k1 = resolve_key(tag, tr[1]);
            const auto k2 = resolve_key(tag, tr[2]);
            if (k0 == k1 || k1 == k2 || k0 == k2)
              continue;
            out[std::size_t(w++)] = tr;
          }
        });
    tris = std::move(out);
    ranges = std::move(new_ranges);
  }

  auto consolidate_merges(tf::buffer<merge_t> &incoming) -> bool {
    if (incoming.size() == 0)
      return false;
    const std::size_t before = _merge.size();
    for (auto &m : incoming)
      _merge.push_back(m);
    incoming.clear();
    // union, not keep-one: A→B and A→C also prove B≡C, so a dropped
    // duplicate re-enters as C→B until the classes are single-rooted
    // (keys strictly decrease, so this terminates)
    tf::buffer<merge_t> proven;
    for (bool grew = true; grew;) {
      grew = false;
      std::sort(_merge.begin(), _merge.end(),
                [](const merge_t &a, const merge_t &b) {
                  if (a.from != b.from)
                    return a.from < b.from;
                  return a.to_key < b.to_key;
                });
      proven.clear();
      std::size_t w = 0;
      for (std::size_t i = 0; i < _merge.size(); ++i) {
        if (w != 0 && _merge[i].from == _merge[w - 1].from) {
          if (_merge[i].to_key != _merge[w - 1].to_key)
            proven.push_back(
                {_merge[i].to_key, _merge[w - 1].to_key, _merge[w - 1].to});
          continue;
        }
        _merge[w++] = _merge[i];
      }
      _merge.erase_till_end(_merge.begin() + std::ptrdiff_t(w));
      for (auto &m : proven)
        _merge.push_back(m);
      grew = proven.size() != 0;
    }
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &m : _merge) {
        auto it = std::lower_bound(
            _merge.begin(), _merge.end(), m.to_key,
            [](const merge_t &x, const std::array<Index, 2> &k) {
              return x.from < k;
            });
        if (it != _merge.end() && it->from == m.to_key &&
            it->to_key != m.from) {
          m.to = it->to;
          m.to_key = it->to_key;
          changed = true;
        }
      }
    }
    return _merge.size() != before;
  }

  template <typename Walk>
  auto count_walk_splits(Index tag, const Walk &walk) const -> Index {
    Index n = 0;
    const std::size_t m = walk.size();
    for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
      auto [s0, s1] =
          splits_of(edge_key(id_of(tag, walk[j]), id_of(tag, walk[i])));
      n += s1 - s0;
    }
    return n;
  }

  auto count_region_steiners(Index li) const -> Index {
    auto lo = std::lower_bound(
        _steiners.begin(), _steiners.end(), li,
        [](const std::array<Index, 2> &a, Index b) { return a[0] < b; });
    auto hi = lo;
    while (hi != _steiners.end() && (*hi)[0] == li)
      ++hi;
    return Index(hi - lo);
  }

  /// Per-region (boundary splits, persisted steiners) — unchanged
  /// means pass 1 would reproduce itself.
  template <typename Fr>
  auto region_fingerprint(const Fr &fr, Index li) const
      -> std::array<Index, 2> {
    const auto d = fr.descriptors()[li];
    Index splits = count_walk_splits(d.tag, fr.loops()[li]);
    for (auto h : fr.loop_holes()[li])
      splits += count_walk_splits(d.tag, fr.holes()[h]);
    return {splits, count_region_steiners(li)};
  }

  template <typename Sc>
  auto inject_persisted_steiners(Index li, Sc &sc) const -> void {
    auto lo = std::lower_bound(
        _steiners.begin(), _steiners.end(), li,
        [](const std::array<Index, 2> &a, Index b) { return a[0] < b; });
    for (auto it = lo; it != _steiners.end() && (*it)[0] == li; ++it) {
      const Index id = (*it)[1];
      const auto &p3 = extra_points[std::size_t(id - n_ig_created)];
      sc.ihm.kept_ids().push_back(
          vertex_t{source::created, id, {0, tf::topo_type::face}});
      sc.pts.push_back({p3[sc.ax0], p3[sc.ax1]});
      sc.pos_set.push_back(0);
    }
  }

  /// Appends the [b, e) block of _promoted_steiners into the scratch;
  /// the promote pass precomputes per-candidate ranges so the parallel
  /// task stays lookup-free.
  template <typename Sc>
  auto inject_promoted_steiners(Index b, Index e, Sc &sc) const -> void {
    for (auto it = _promoted_steiners.begin() + std::ptrdiff_t(b);
         it != _promoted_steiners.begin() + std::ptrdiff_t(e); ++it) {
      const Index id = (*it)[2];
      const auto &p3 = extra_points[std::size_t(id - n_ig_created)];
      sc.ihm.kept_ids().push_back(
          vertex_t{source::created, id, {0, tf::topo_type::face}});
      sc.pts.push_back({p3[sc.ax0], p3[sc.ax1]});
      sc.pos_set.push_back(0);
    }
  }

  /// The on-edge fusion ball, shared by recovery registration and the
  /// batch sweep: a split within one lattice param of an existing one
  /// on the same edge IS that split (same edge = same carriers, so
  /// fusing is local and identity-clean; originals always win
  /// identity, so no original id is ever removed by fusion).
  auto has_adjacent_split(const ekey_t &key, Index num) const -> bool {
    auto [s0, s1] = splits_of(key);
    for (Index t = s0; t < s1; ++t) {
      const Index d = _splits[std::size_t(t)].param - num;
      if (d <= Index(1) && d >= Index(-1))
        return true;
    }
    return false;
  }

  auto splits_of(const ekey_t &key) const -> std::pair<Index, Index> {
    struct cmp {
      auto operator()(const split_t &s, const ekey_t &k) const -> bool {
        return s.edge < k;
      }
      auto operator()(const ekey_t &k, const split_t &s) const -> bool {
        return k < s.edge;
      }
    };
    auto r = std::equal_range(_splits.begin(), _splits.end(), key, cmp{});
    return {Index(r.first - _splits.begin()),
            Index(r.second - _splits.begin())};
  }

  /// The recovery loop: discover with intersections allowed, materialize
  /// once, broadcast to every carrier, retry preserving; repeat until
  /// clear or the round cap (survivors stay in `failed`, loud).
  template <typename ApplyToFace, typename GetMeshPoint, typename DeadLoops>
  auto recover(const tf::face_regions<Index, Int> &fr,
               const tf::intersection_graph<Index, Int> &ig,
               const ApplyToFace &apply_to_face,
               const GetMeshPoint &get_mesh_point,
               const DeadLoops &dead_loops) -> void {
    auto is_dead = [&](std::size_t li) -> bool {
      return dead_loops.size() != 0 && bool(dead_loops[li]);
    };
    auto descs = fr.descriptors();
    auto loops = fr.loops();
    auto loop_holes = fr.loop_holes();
    auto holes = fr.holes();
    auto ipts = ig.points();

    auto get_point_g = [&](Index tag, const vertex_t &v) {
      return point_of(tag, v, ipts, get_mesh_point);
    };

    auto touches_merge = [&](Index tag, const auto &walk) {
      for (const auto &v : walk)
        if (resolve_key(tag, v) != id_of(tag, v))
          return true;
      return false;
    };
    bool merges_fresh = _merge.size() != 0;

    for (int round = 0;
         round < k_max_rounds && (failed.size() || merges_fresh); ++round) {
      const std::size_t n_splits_before = _splits.size();

      // DISCOVERY in parallel: preserving re-assembly + discovery CDT
      // emit crossing records only — materialization stays out of the
      // threads so the sequenced stream hands out extra-point ids in
      // the exact order the serial loop did.
      struct disco_local_t {
        scratch_t sc;
        tf::buffer<std::array<vertex_t, 3>> scrap;
        tf::buffer<rec_proposal_t> proposals;
        tf::buffer<std::array<vertex_t, 2>> parent_edges;
      };
      tf::buffer<rec_proposal_t> proposals;
      tf::buffer<std::array<vertex_t, 2>> parent_edges;
      tf::buffer<merge_t> pending;
      auto disco_task = [&](auto &&range, disco_local_t &local) {
        auto apply_to_face_f = apply_to_face;
        auto get_mesh_point_f = get_mesh_point;
        for (auto li : range) {
          const auto &desc = descs[li];
          // assemble the (split-augmented) inputs; the preserving run
          // refuses again but leaves sc.cons/cons_verts/pts populated
          local.scrap.clear();
          run_loop(fr, desc, loops[li], loop_holes[li], apply_to_face_f,
                   get_mesh_point_f, ipts, local.sc, local.scrap, true);
          discover_crossings(local.sc, desc.tag, local.proposals,
                             local.parent_edges);
        }
      };
      auto disco_agg = [&](const disco_local_t &local, const tf::none_t &) {
        tf::core::append(local.proposals, proposals);
        tf::core::append(local.parent_edges, parent_edges);
        tf::core::append(local.sc.merges, pending);
      };
      tf::blocked_reduce_sequenced_aggregate(tf::make_range(failed), tf::none,
                                             disco_local_t{}, disco_task,
                                             disco_agg);

      // materialize serially over the ordered stream
      materialize_crossings(proposals, parent_edges, get_point_g);
      consolidate_merges(pending);
      if (_splits.size() == n_splits_before && !merges_fresh)
        break; // nothing new to consume — survivors are terminal
      std::sort(_splits.begin(), _splits.end(),
                [](const split_t &x, const split_t &y) {
                  if (x.edge != y.edge)
                    return x.edge < y.edge;
                  return x.param < y.param;
                });
      _splits.erase_till_end(
          std::unique(_splits.begin(), _splits.end(),
                      [](const split_t &x, const split_t &y) {
                        return x.edge == y.edge && x.key == y.key;
                      }));

      // MARK in parallel: every loop whose walks touch a split edge; a
      // dead loop never re-triangulates — it follows its survivor's
      // range. Serial compaction keeps loop order.
      tf::buffer<char> touched_flags;
      touched_flags.allocate(std::size_t(loops.size()));
      tf::parallel_for_each(
          tf::make_sequence_range(Index(loops.size())), [&](Index li) {
            touched_flags[std::size_t(li)] = char(0);
            if (descs[li].tag == Index(-1) || loops[li].size() < 3 ||
                is_dead(std::size_t(li)))
              return;
            bool touched = false;
            const Index ltag = descs[li].tag;
            auto scan = [&](const auto &walk) {
              const auto m = walk.size();
              for (std::size_t i = 0, j = m - 1; i < m && !touched;
                   j = i++) {
                auto [s0, s1] = splits_of(
                    edge_key(id_of(ltag, walk[j]), id_of(ltag, walk[i])));
                touched = s0 != s1;
              }
            };
            scan(loops[li]);
            for (auto h : loop_holes[li]) {
              if (touched)
                break;
              scan(holes[h]);
            }
            if (!touched && merges_fresh) {
              touched = touches_merge(ltag, loops[li]);
              for (auto h : loop_holes[li]) {
                if (touched)
                  break;
                touched = touches_merge(ltag, holes[h]);
              }
            }
            touched_flags[std::size_t(li)] = char(touched);
          });
      tf::buffer<Index> dirty;
      for (Index li = 0; li < Index(loops.size()); ++li)
        if (touched_flags[std::size_t(li)])
          dirty.push_back(li);
      for (auto li : failed)
        dirty.push_back(li);
      std::sort(dirty.begin(), dirty.end());
      dirty.erase_till_end(std::unique(dirty.begin(), dirty.end()));

      // RETRY preserving in parallel; sequenced aggregation rebuilds
      // the affected loops' triangle blocks in dirty order
      struct retry_local_t {
        scratch_t sc;
        tf::buffer<std::array<vertex_t, 3>> out;
        tf::buffer<Index> counts;
        tf::buffer<Index> failed;
      };
      tf::buffer<std::array<vertex_t, 3>> new_tris;
      tf::buffer<Index> retry_counts;
      tf::buffer<Index> still_failed;
      tf::buffer<merge_t> retry_pending;
      auto retry_task = [&](auto &&range, retry_local_t &local) {
        auto apply_to_face_f = apply_to_face;
        auto get_mesh_point_f = get_mesh_point;
        for (auto li : range) {
          const std::size_t before = local.out.size();
          const bool ok =
              run_loop(fr, descs[li], loops[li], loop_holes[li],
                       apply_to_face_f, get_mesh_point_f, ipts, local.sc,
                       local.out, true);
          if (!ok) {
            local.out.erase_till_end(local.out.begin() +
                                     std::ptrdiff_t(before));
            local.failed.push_back(li);
          }
          local.counts.push_back(Index(local.out.size() - before));
        }
      };
      auto retry_agg = [&](const retry_local_t &local, const tf::none_t &) {
        tf::core::append(local.out, new_tris);
        tf::core::append(local.counts, retry_counts);
        tf::core::append(local.failed, still_failed);
        tf::core::append(local.sc.merges, retry_pending);
      };
      tf::blocked_reduce_sequenced_aggregate(tf::make_range(dirty), tf::none,
                                             retry_local_t{}, retry_task,
                                             retry_agg);
      tf::buffer<std::array<Index, 2>> new_ranges;
      new_ranges.allocate(dirty.size());
      Index acc = 0;
      for (std::size_t d = 0; d < dirty.size(); ++d) {
        new_ranges[d] = {acc, acc + retry_counts[d]};
        acc += retry_counts[d];
      }
      splice(dirty, new_ranges, new_tris);
      failed = std::move(still_failed);
      merges_fresh = consolidate_merges(retry_pending);
    }
  }

  template <typename GetPoint>
  auto register_split(Index tag, const std::array<vertex_t, 2> &edge,
                      const vertex_t &v, const GetPoint &get_point,
                      bool named = false) -> void {
    const auto k0 = id_of(tag, edge[0]);
    const auto k1 = id_of(tag, edge[1]);
    auto key = edge_key(k0, k1);
    const auto &lo = k0 < k1 ? edge[0] : edge[1];
    const auto &hi = k0 < k1 ? edge[1] : edge[0];
    const auto kv = id_of(tag, v);
    if (kv == k0 || kv == k1)
      return;
    auto a = get_point(tag, lo);
    auto b = get_point(tag, hi);
    auto p = get_point(tag, v);
    T2 param(0);
    for (int c = 0; c < 3; ++c)
      param += T2(T1(p[c]) - T1(a[c])) * T2(T1(b[c]) - T1(a[c]));
    // rounding can push the materialized point past an endpoint; the
    // parameter is clamped so a carrier never extrapolates off its
    // edge — the position lands on the endpoint and the weld retires
    // the identity (originals win)
    Index num = Index(0);
    if (T2(0) < param) {
      T2 t_max(0);
      for (int c = 0; c < 3; ++c) {
        const T1 d = T1(b[c]) - T1(a[c]);
        t_max += T2(d) * T2(d);
      }
      if (t_max <= param)
        num = Index(1) << k_param_bits;
      else
        num = static_cast<Index>(tf::exact::div_round(
            param * T2(T1(1) << k_param_bits), t_max));
    }
    if (has_adjacent_split(key, num))
      return;
    _splits.push_back({key, num, kv, v, char(named)});
  }

  /// Materialize the dyadic point `num/2^depth` along a physical edge —
  /// a convex combination of the endpoints, so it lies on that edge in
  /// every face that carries it, and every carrier computes the same
  /// lattice point.
  template <typename GetMeshPoint, typename Ipts>
  auto register_edge_point(Index tag, const std::array<vertex_t, 2> &edge,
                           std::uint8_t num, std::uint8_t depth,
                           const GetMeshPoint &get_mesh_point,
                           const Ipts &ipts) -> void {
    auto pt = [&](const vertex_t &v) -> tf::point<Int, 3> {
      if (v.source == source::created) {
        if (v.id >= n_ig_created)
          return extra_points[std::size_t(v.id - n_ig_created)];
        return ipts[v.id];
      }
      return get_mesh_point(tag, v.id);
    };
    const auto k0 = id_of(tag, edge[0]);
    const auto k1 = id_of(tag, edge[1]);
    const auto &lo = k0 < k1 ? edge[0] : edge[1];
    const auto &hi = k0 < k1 ? edge[1] : edge[0];
    const bool fwd = k0 < k1;
    const T1 den = T1(1) << depth;
    const T1 nu = fwd ? T1(num) : den - T1(num);
    auto a = pt(lo);
    auto b = pt(hi);
    tf::point<Int, 3> p;
    for (int c = 0; c < 3; ++c)
      p[c] = static_cast<Int>(tf::exact::div_round(
          T1(a[c]) * (den - nu) + T1(b[c]) * nu, den));
    auto v = vertex_t{source::created,
                      n_ig_created + Index(extra_points.size()),
                      {0, tf::topo_type::edge}};
    extra_points.push_back(p);
    register_split(tag, edge, v, [&](Index, const vertex_t &w) { return pt(w); });
  }

  auto sort_splits() -> void {
    std::sort(_splits.begin(), _splits.end(),
              [](const split_t &x, const split_t &y) {
                if (x.edge != y.edge)
                  return x.edge < y.edge;
                return x.param < y.param;
              });
    _splits.erase_till_end(
        std::unique(_splits.begin(), _splits.end(),
                    [](const split_t &x, const split_t &y) {
                      return x.edge == y.edge && x.key == y.key;
                    }));
  }

  /// Re-triangulate every live region against the current split table,
  /// in the shape the rest of the pipeline uses: block-local work,
  /// ordered aggregation. With a refiner, each region's final
  /// tessellation is the refined one — its boundary frozen (the splits
  /// are already agreed globally) and the points it inserts inside
  /// become created vertices of that face. Those are face-local, so a
  /// chunk numbers them locally and the aggregator rebases.
  template <typename ApplyToFace, typename GetMeshPoint, typename Ipts,
            typename DeadLoops, typename Refiner = tf::none_t>
  auto retriangulate_all(const tf::face_regions<Index, Int> &fr,
                         const ApplyToFace &apply_to_face,
                         const GetMeshPoint &get_mesh_point, const Ipts &ipts,
                         scratch_t &, const DeadLoops &dead_loops,
                         Refiner *rf = nullptr,
                         const tf::cdt_refine_config *config = nullptr,
                         const tf::buffer<Index> *steiner_off = nullptr,
                         Index steiner_base = Index(0),
                         tf::buffer<merge_t> *pending = nullptr) -> void {
    auto descs = fr.descriptors();
    auto loops = fr.loops();
    auto loop_holes = fr.loop_holes();

    struct local_t {
      scratch_t sc;
      Refiner rf;
      tf::buffer<Index> to_vertex;
      tf::buffer<bool> frozen;
      tf::buffer<std::array<vertex_t, 3>> out;
      tf::buffer<tf::point<Int, 3>> interior;
      tf::buffer<Index> counts;
      tf::buffer<Index> failed;
    };
    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (auto &&[li_z, desc, loop, hole_ids] : range) {
        const std::size_t before = local.out.size();
        bool ok = true;
        // dead members emit nothing here — the alias re-assert points
        // their range at the survivor's block, as in the build path
        if (desc.tag != Index(-1) && loop.size() >= 3 &&
            !(dead_loops.size() != 0 && bool(dead_loops[li_z]))) {
          const bool refined_run = [&] {
            if constexpr (std::is_same_v<Refiner, tf::none_t>)
              return false;
            else
              return rf != nullptr;
          }();
          ok = run_loop(fr, desc, loop, hole_ids, apply_to_face_f,
                        get_mesh_point_f, ipts, local.sc, local.out, true,
                        /*assemble_only=*/refined_run);
          if constexpr (!std::is_same_v<Refiner, tf::none_t>) {
            if (ok && rf != nullptr) {
              local.out.erase_till_end(local.out.begin() +
                                       std::ptrdiff_t(before));
              if (steiner_off != nullptr) {
                inject_persisted_steiners(Index(li_z), local.sc);
                // pass-1 interiors re-enter as plain inputs with their
                // pre-materialized identities
                for (Index k = (*steiner_off)[std::size_t(li_z)];
                     k < (*steiner_off)[std::size_t(li_z) + 1]; ++k) {
                  const auto &p3 =
                      extra_points[std::size_t(steiner_base + k)];
                  local.sc.ihm.kept_ids().push_back(
                      vertex_t{source::created,
                               n_ig_created + steiner_base + k,
                               {0, tf::topo_type::face}});
                  local.sc.pts.push_back(
                      {p3[local.sc.ax0], p3[local.sc.ax1]});
                  local.sc.pos_set.push_back(0);
                }
              }
              if (!emit_refined(local.sc, desc.tag, local.rf, *config,
                                local.to_vertex, local.frozen, local.out,
                                &local.interior))
                ok = run_loop(fr, desc, loop, hole_ids, apply_to_face_f,
                              get_mesh_point_f, ipts, local.sc, local.out,
                              true);
            }
          }
        }
        if (!ok) {
          local.out.erase_till_end(local.out.begin() + std::ptrdiff_t(before));
          local.failed.push_back(Index(li_z));
        }
        local.counts.push_back(Index(local.out.size() - before));
      }
    };
    tf::buffer<std::array<vertex_t, 3>> out;
    tf::buffer<Index> counts;
    failed.clear();
    // Interiors accumulate locally and join extra_points only after
    // the reduce: tasks read extra_points concurrently (split
    // vertices), so the aggregator must never reallocate it.
    tf::buffer<tf::point<Int, 3>> interior_all;
    const Index interior_pre = Index(extra_points.size());
    auto agg = [&](const local_t &local, const tf::none_t &) {
      // this chunk's interior points join the table; its triangles
      // reference them by local number, rebased here
      const Index base = interior_pre + Index(interior_all.size());
      tf::core::append(local.interior, interior_all);
      const std::size_t first = out.size();
      tf::core::append(local.out, out);
      if (local.interior.size())
        for (std::size_t i = first; i < out.size(); ++i)
          for (int c = 0; c < 3; ++c) {
            auto &v = out[i][std::size_t(c)];
            if (v.source == source::created &&
                std::make_signed_t<Index>(v.id) < 0)
              v.id = n_ig_created + base + (-v.id - 1);
          }
      tf::core::append(local.counts, counts);
      tf::core::append(local.failed, failed);
      if (pending != nullptr)
        tf::core::append(local.sc.merges, *pending);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::zip(tf::make_sequence_range(std::size_t(loops.size())), descs,
                loops, loop_holes),
        tf::none, local_t{}, task, agg);

    tf::core::append(interior_all, extra_points);
    tris = std::move(out);
    ranges.clear();
    ranges.allocate(std::size_t(loops.size()));
    Index acc = 0;
    for (std::size_t li = 0; li < std::size_t(loops.size()); ++li) {
      ranges[li] = {acc, acc + counts[li]};
      acc += counts[li];
    }
    n_failed = static_cast<Index>(failed.size());
  }

  /// Refine the assembled region (boundary frozen) and emit its
  /// triangles, lifting the interior points the refiner inserts into
  /// created vertices. Returns false if the refiner declines, so the
  /// caller can fall back to the plain triangulation.
  template <typename Refiner>
  auto emit_refined(scratch_t &sc, Index tag, Refiner &rf,
                    const tf::cdt_refine_config &config,
                    tf::buffer<Index> &to_vertex, tf::buffer<bool> &frozen,
                    tf::buffer<std::array<vertex_t, 3>> &out,
                    tf::buffer<tf::point<Int, 3>> *sink = nullptr) -> bool {
    auto edges = tf::make_edges(tf::make_blocked_range<2>(
        tf::make_range(sc.cons.data(), sc.cons.size())));
    frozen.allocate(edges.size());
    std::fill(frozen.begin(), frozen.end(), false);
    rf.clear();
    if (!rf.build(tf::make_points(sc.pts), edges,
                  tf::make_constant_range(true, edges.size()),
                  tf::make_range(frozen), config))
      return false;

    // output point -> vertex: inputs through the refiner's index map,
    // inserted points become new created ids on this face's plane
    const auto &kept = sc.ihm.kept_ids();
    const auto &fmap = rf.index_map().f();
    const Index n_out = Index(rf.points().size());
    to_vertex.allocate(std::size_t(n_out));
    std::fill(to_vertex.begin(), to_vertex.end(), Index(-1));
    bool welded = false;
    for (Index i = 0; i < Index(kept.size()); ++i) {
      const Index o = fmap[std::size_t(i)];
      if (o < 0 || n_out <= o)
        return false;
      auto &tv = to_vertex[std::size_t(o)];
      if (tv == Index(-1)) {
        tv = i;
      } else if (id_of(tag, kept[std::size_t(i)]) ==
                 id_of(tag, kept[std::size_t(tv)])) {
        // map-free duplicate of one identity: not a weld, either
        // occurrence serves
      } else {
        // the refiner welded projected-coincident inputs: the
        // canonical (smallest-identity) representative survives —
        // first-wins would let carriers disagree across faces
        welded = true;
        if (id_of(tag, kept[std::size_t(i)]) <
            id_of(tag, kept[std::size_t(tv)]))
          tv = i;
      }
    }
    if (welded) {
      // report globally: every other carrier must retire the same
      // identities in favour of the same survivor
      for (Index i = 0; i < Index(kept.size()); ++i) {
        const Index r = to_vertex[std::size_t(fmap[std::size_t(i)])];
        if (r != i &&
            id_of(tag, kept[std::size_t(i)]) !=
                id_of(tag, kept[std::size_t(r)]))
          sc.merges.push_back({id_of(tag, kept[std::size_t(i)]),
                               id_of(tag, kept[std::size_t(r)]),
                               kept[std::size_t(r)]});
      }
    }
    const std::size_t extra_before =
        sink != nullptr ? sink->size() : extra_points.size();
    auto rpts = rf.points();
    auto interior_vertex = [&](Index o) -> vertex_t {
      auto p3 = lift_to_plane(sc, rpts[std::size_t(o)]);
      if (sink != nullptr) { // numbered per chunk; the aggregator rebases
        auto v = vertex_t{source::created, Index(-Index(sink->size()) - 1),
                          {0, tf::topo_type::face}};
        sink->push_back(p3);
        return v;
      }
      auto v = vertex_t{source::created,
                        n_ig_created + Index(extra_points.size()),
                        {0, tf::topo_type::face}};
      extra_points.push_back(p3);
      return v;
    };
    tf::buffer<vertex_t> inserted;
    inserted.allocate(std::size_t(n_out));
    tf::buffer<char> has_inserted;
    has_inserted.allocate(std::size_t(n_out));
    std::fill(has_inserted.begin(), has_inserted.end(), char(0));

    auto labels = rf.region_labels();
    const std::size_t out_before = out.size();
    for (Index f = 0; f < rf.n_faces(); ++f) {
      if ((labels[std::size_t(f)] & 1) == 0)
        continue;
      auto tri = rf.face(f);
      std::array<vertex_t, 3> vs;
      for (int c = 0; c < 3; ++c) {
        const Index o = tri[std::size_t(c)];
        if (o < 0 || n_out <= o) {
          if (sink != nullptr)
            sink->erase_till_end(sink->begin() + std::ptrdiff_t(extra_before));
          else
            extra_points.erase_till_end(extra_points.begin() +
                                        std::ptrdiff_t(extra_before));
          out.erase_till_end(out.begin() + std::ptrdiff_t(out_before));
          return false;
        }
        const Index in = to_vertex[std::size_t(o)];
        if (in != Index(-1)) {
          vs[std::size_t(c)] = kept[std::size_t(in)];
        } else {
          if (!has_inserted[std::size_t(o)]) {
            inserted[std::size_t(o)] = interior_vertex(o);
            has_inserted[std::size_t(o)] = 1;
          }
          vs[std::size_t(c)] = inserted[std::size_t(o)];
        }
      }
      // the refiner's faces come out mirrored relative to sc.tri's —
      // the flip rule is therefore inverted here (signed-cover
      // verified against the stock stream)
      out.push_back(sc.flip ? vs
                            : std::array<vertex_t, 3>{vs[0], vs[2], vs[1]});
    }
    return true;
  }

  /// Lift a 2D point of this face's projection back onto its plane,
  /// using the plane the face's own vertices define.
  auto lift_to_plane(const scratch_t &sc, const tf::point<Int, 2> &p) const
      -> tf::point<Int, 3> {
    tf::point<Int, 3> out;
    const int ax0 = sc.ax0, ax1 = sc.ax1, ax2 = 3 - ax0 - ax1;
    out[ax0] = p[0];
    out[ax1] = p[1];
    // plane through the face's first three distinct projected points
    const auto &n = sc.plane_n;
    // n . x = n . a  ->  x[ax2] = (n.a - n0*p0 - n1*p1) / n2
    const T2 rhs = sc.plane_d - n[std::size_t(ax0)] * T2(p[0]) -
                   n[std::size_t(ax1)] * T2(p[1]);
    const T2 den = n[std::size_t(ax2)];
    out[ax2] = den == T2(0)
                   ? sc.plane_fallback
                   : static_cast<Int>(tf::exact::div_round(rhs, den));
    return out;
  }

  /// Pull every uncut face that now carries a split into the loop set.
  /// Its walk is the original triangle — `run_loop` inserts the split
  /// vertices itself, each positioned on this face's own edge, so the
  /// walk stays simple.
  template <typename ApplyToForm, typename ApplyToFace, typename GetMeshPoint,
            typename Ipts, typename Refiner = tf::none_t>
  auto promote(const tf::face_regions<Index, Int> &fr, Index n_tags,
               const ApplyToForm &apply_to_form,
               const ApplyToFace &apply_to_face,
               const GetMeshPoint &get_mesh_point, const Ipts &ipts,
               Refiner *rf = nullptr,
               const tf::cdt_refine_config *config = nullptr,
               tf::buffer<merge_t> *pending = nullptr) -> void {
    _promoted_interior_of.clear();
    if (_original_splits.size() == 0)
      return;
    auto descs = fr.descriptors();
    tf::core::std_vector<tf::buffer<char>> cut(static_cast<std::size_t>(n_tags));
    for (std::size_t t = 0; t < std::size_t(n_tags); ++t) {
      apply_to_form(Index(t), [&](const auto &form) {
        cut[t].allocate(form.faces().size());
      });
      std::fill(cut[t].begin(), cut[t].end(), char(0));
    }
    for (Index l = 0; l < Index(descs.size()); ++l)
      if (descs[l].tag != Index(-1))
        cut[std::size_t(descs[l].tag)][std::size_t(descs[l].object)] = 1;
    for (Index t = 0; t < n_tags; ++t)
      for (auto d : fr.deleted(t))
        cut[std::size_t(t)][std::size_t(d)] = 1;

    tf::buffer<std::array<Index, 2>> candidates;
    for (const auto &os : _original_splits) {
      apply_to_form(os.tag, [&](const auto &form) {
        for (auto f_id : form.face_membership()[os.u]) {
          if (cut[std::size_t(os.tag)][std::size_t(f_id)])
            continue;
          auto face = form.faces()[f_id];
          const Index n = Index(face.size());
          for (Index e = 0; e < n; ++e) {
            const Index a = Index(face[std::size_t(e)]);
            const Index b = Index(face[std::size_t((e + 1) % n)]);
            if ((a == os.u && b == os.v) || (a == os.v && b == os.u)) {
              candidates.push_back({os.tag, Index(f_id)});
              break;
            }
          }
        }
      });
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase_till_end(
        std::unique(candidates.begin(), candidates.end()));
    if (candidates.size() == 0)
      return;
    // both lists sorted by (tag, object): one merge walk gives every
    // candidate its persisted-steiner block
    tf::buffer<std::array<Index, 2>> cand_steiner_ranges;
    cand_steiner_ranges.allocate(candidates.size());
    {
      std::size_t t = 0;
      for (std::size_t k = 0; k < candidates.size(); ++k) {
        while (t < _promoted_steiners.size() &&
               (_promoted_steiners[t][0] < candidates[k][0] ||
                (_promoted_steiners[t][0] == candidates[k][0] &&
                 _promoted_steiners[t][1] < candidates[k][1])))
          ++t;
        const Index b = Index(t);
        while (t < _promoted_steiners.size() &&
               _promoted_steiners[t][0] == candidates[k][0] &&
               _promoted_steiners[t][1] == candidates[k][1])
          ++t;
        cand_steiner_ranges[k] = {b, Index(t)};
      }
    }

    // Promoted faces triangulate independently — the same parallel
    // shape as retriangulate_all, interiors rebased at aggregation.
    struct plocal_t {
      scratch_t sc;
      Refiner rf_l;
      tf::buffer<vertex_t> walk;
      tf::buffer<Index> no_holes;
      tf::buffer<Index> to_vertex;
      tf::buffer<bool> frozen;
      tf::buffer<std::array<vertex_t, 3>> out;
      tf::buffer<tf::point<Int, 3>> interior;
      tf::buffer<Index> counts;
      tf::buffer<Index> int_counts;
      tf::buffer<char> ok_flags;
    };
    auto ptask = [&](auto &&range, plocal_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (auto &&[ck, c] : range) {
        desc_t d{c[0], c[1]};
        local.walk.clear();
        apply_to_face_f(c[0], c[1], [&](const auto &face) {
          for (std::size_t i = 0; i < face.size(); ++i)
            local.walk.push_back(vertex_t{source::original, Index(face[i]),
                                          {short(i), tf::topo_type::vertex}});
        });
        const std::size_t before = local.out.size();
        const std::size_t before_int = local.interior.size();
        const bool refined_run = [&] {
          if constexpr (std::is_same_v<Refiner, tf::none_t>)
            return false;
          else
            return rf != nullptr;
        }();
        bool ok = run_loop(fr, d, tf::make_range(local.walk),
                           tf::make_range(local.no_holes), apply_to_face_f,
                           get_mesh_point_f, ipts, local.sc, local.out, true,
                           /*assemble_only=*/refined_run);
        if constexpr (!std::is_same_v<Refiner, tf::none_t>) {
          if (ok && rf != nullptr) {
            inject_promoted_steiners(cand_steiner_ranges[std::size_t(ck)][0],
                                     cand_steiner_ranges[std::size_t(ck)][1],
                                     local.sc);
            // a promoted face is refined like any other region — the
            // same patching, the same quality
            local.out.erase_till_end(local.out.begin() +
                                     std::ptrdiff_t(before));
            if (!emit_refined(local.sc, d.tag, local.rf_l, *config,
                              local.to_vertex, local.frozen, local.out,
                              &local.interior))
              ok = run_loop(fr, d, tf::make_range(local.walk),
                            tf::make_range(local.no_holes), apply_to_face_f,
                            get_mesh_point_f, ipts, local.sc, local.out, true);
          }
        }
        if (!ok)
          local.out.erase_till_end(local.out.begin() + std::ptrdiff_t(before));
        local.counts.push_back(Index(local.out.size() - before));
        local.int_counts.push_back(Index(local.interior.size() - before_int));
        local.ok_flags.push_back(char(ok));
      }
    };
    tf::buffer<tf::point<Int, 3>> interior_all;
    const Index interior_pre = Index(extra_points.size());
    _promoted_interior_base = interior_pre;
    auto pagg = [&](const plocal_t &local, const tf::none_t &) {
      const Index base = interior_pre + Index(interior_all.size());
      tf::core::append(local.interior, interior_all);
      const std::size_t first = tris.size();
      tf::core::append(local.out, tris);
      if (local.interior.size())
        for (std::size_t i = first; i < tris.size(); ++i)
          for (int ci = 0; ci < 3; ++ci) {
            auto &v = tris[i][std::size_t(ci)];
            if (v.source == source::created &&
                std::make_signed_t<Index>(v.id) < 0)
              v.id = n_ig_created + base + (-v.id - 1);
          }
      Index acc = Index(first);
      for (std::size_t i = 0; i < local.counts.size(); ++i) {
        if (!local.ok_flags[i])
          failed.push_back(Index(ranges.size()));
        ranges.push_back({acc, acc + local.counts[i]});
        acc += local.counts[i];
      }
      if (pending != nullptr)
        tf::core::append(local.sc.merges, *pending);
      tf::core::append(local.int_counts, _promoted_interior_of);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::zip(tf::make_sequence_range(candidates.size()),
                tf::make_range(candidates)),
        tf::none, plocal_t{}, ptask, pagg);
    tf::core::append(interior_all, extra_points);
    for (const auto &c : candidates)
      _promoted.push_back(desc_t{c[0], c[1]});
    n_failed = static_cast<Index>(failed.size());
  }

  /// Project the split table onto the original mesh edges — the only
  /// ones an uncut face can share. Order along the edge is the table's
  /// own (sorted by parameter from the min-identity endpoint).
  auto collect_original_edge_splits() -> void {
    _original_splits.clear();
    for (const auto &sp : _splits) {
      const auto &ka = sp.edge[0];
      const auto &kb = sp.edge[1];
      if (is_created(ka) || is_created(kb) || ka[0] != kb[0])
        continue; // an endpoint is a cut point: no uncut face shares it
      _original_splits.push_back({ka[0], ka[1], kb[1], sp.vertex});
    }
  }

  /// Replace the affected loops' triangle blocks and rebuild the global
  /// stream once.
  auto splice(const tf::buffer<Index> &dirty,
              const tf::buffer<std::array<Index, 2>> &new_ranges,
              const tf::buffer<std::array<vertex_t, 3>> &new_tris) -> void {
    tf::buffer<std::array<vertex_t, 3>> merged;
    tf::buffer<std::array<Index, 2>> merged_ranges;
    merged_ranges.allocate(ranges.size());
    std::size_t d = 0;
    Index acc = 0;
    for (std::size_t li = 0; li < ranges.size(); ++li) {
      const bool replaced = d < dirty.size() && dirty[d] == Index(li);
      const auto r = replaced ? new_ranges[d] : ranges[li];
      const auto &src = replaced ? new_tris : tris;
      for (Index t = r[0]; t < r[1]; ++t)
        merged.push_back(src[std::size_t(t)]);
      merged_ranges[li] = {acc, acc + (r[1] - r[0])};
      acc += r[1] - r[0];
      d += replaced;
    }
    tris = std::move(merged);
    ranges = std::move(merged_ranges);
  }

  auto copy_deleted(const tf::face_regions<Index, Int> &fr) -> void {
    auto offs = fr.deleted_offsets();
    const std::size_t n = offs.size();
    _deleted_offsets.allocate(n);
    for (std::size_t i = 0; i < n; ++i)
      _deleted_offsets[i] = offs[i];
    _deleted.allocate(n ? std::size_t(offs[n - 1]) : 0);
    for (std::size_t t = 0; t + 1 < n; ++t)
      tf::parallel_copy(fr.deleted(static_cast<Index>(t)),
                        tf::slice(tf::make_range(_deleted),
                                  std::size_t(offs[t]),
                                  std::size_t(offs[t + 1])));
  }

  Index _n_tags = 0;
  tf::buffer<split_t> _splits;
  tf::buffer<original_edge_split> _original_splits;
  tf::buffer<desc_t> _promoted;
  tf::buffer<merge_t> _merge; // sorted by `from`, transitively closed
  // persisted interior steiners, (region, created id) sorted by region:
  // every refine re-injects them so iterated refinement sees current
  // state and their points stay referenced
  tf::buffer<std::array<Index, 2>> _steiners;
  // refine convergence: per-region (boundary splits, persisted steiners)
  // recorded at refine end; an unchanged region skips pass-1 quality —
  // repeat refines reach an exact fixed point
  tf::buffer<std::array<Index, 2>> _region_fingerprints;
  float _refined_quality = -1.f;
  // promoted-face interiors persist the same way, keyed (tag, object, id)
  // sorted; re-injection keeps iterated refinement convergent
  tf::buffer<std::array<Index, 3>> _promoted_steiners;
  tf::buffer<Index> _promoted_interior_of; // per promoted face, count
  Index _promoted_interior_base = 0;
  bool _aliased = false;
  bool _exposed_owned = false;
  tf::buffer<std::array<vertex_t, 3>> _exposed_triangles;
  tf::buffer<desc_t> _exposed_descriptors;
  tf::buffer<std::array<Index, 2>> _exposed_ranges;
  tf::buffer<Index> _face_offsets;
  tf::buffer<Index> _tag_offsets;
  tf::buffer<Index> _deleted;
  tf::buffer<Index> _deleted_offsets;
};

} // namespace tf::cut
