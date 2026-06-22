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

#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/make_equivalence_class_map.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/hash_map.hpp"
#include "../core/hash_set.hpp"
#include "../core/memory.hpp"
#include "../core/range.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/zip.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../topology/compare_faces.hpp"
#include "../topology/connected_component_labels.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/label_connected_components.hpp"
#include "../topology/structures/compute_face_link_per_edge.hpp"
#include "./face_cuts.hpp"
#include "./partition/make_labels.hpp"
#include "./resolve_face_edge.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

#include <algorithm>
#include <numeric>

namespace tf {

/// @ingroup cut
/// @brief Implicit arrangement of N polygon forms.
///
/// Encodes everything needed to reason about an N-form arrangement
/// without materialising a merged mesh:
///
/// - **Polygon labels per form**: per uncut surface face of each
///   form, the arrangement component it belongs to.
/// - **Loop labels**: per cut loop (in flat `face_cuts` ordering),
///   the arrangement component it belongs to, or `none_label` if the
///   loop was dropped as a coplanar duplicate during build.
/// - **Per-loop per-edge connectivity**: dead loops filtered out
///   on both sides (their entries are empty and no live loop
///   references them).
///
/// Coplanar duplicate loops are detected at build time
/// (`compare_faces` against edge-0 neighbours of different tags) and
/// loops from all but the minimum-tag form are marked dead. The
/// surviving form represents the wall; the others are invisible to
/// every downstream consumer.
///
/// The graph does **not** retain references to the input forms, the
/// `intersection_graph`, or the `face_cuts`. Forms are read during
/// `build()` only.
///
/// @tparam Index The integer index type for vertex / face / loop ids.
template <typename Index> class arrangement_graph {
public:
  /// @brief Component-id label type.
  using label_type = Index;

  /// @brief Sentinel for "no component": face was cut, or loop was
  ///        dropped as a coplanar duplicate.
  static constexpr label_type none_label = label_type(-1);

  /// @brief Default-constructs an empty graph. Use @ref build to
  ///        populate.
  arrangement_graph() = default;

  /// @brief Number of input forms (tags) the graph was built from.
  auto n_tags() const -> Index {
    return static_cast<Index>(_polygon_labels.size());
  }

  /// @brief Total number of distinct arrangement components.
  ///
  /// Components are numbered in `[0, n_components())`. Dropped
  /// (coplanar-duplicate) loops carry `none_label` instead.
  auto n_components() const -> Index { return _n_components; }

  /// @brief Per-face surface component labels for form `tag`.
  ///
  /// `polygon_labels(tag)[face_id]` is the global arrangement
  /// component id of face `face_id` in form `tag`, or `none_label` if
  /// the face was cut by the arrangement (its sub-loops then carry
  /// the labels via @ref loop_labels).
  auto polygon_labels(Index tag) const {
    return tf::make_range(_polygon_labels[static_cast<std::size_t>(tag)]);
  }

  /// @brief Flat per-loop labels across all tags.
  ///
  /// `loop_labels()[loop_id]` is the global arrangement component id
  /// of the loop, or `none_label` if the loop was dropped as a
  /// coplanar duplicate. `loop_id` matches `face_cuts.descriptors()`
  /// ordering.
  auto loop_labels() const { return tf::make_range(_loop_labels); }

  /// @brief Per-tag slice of @ref loop_labels.
  auto loop_labels(Index tag) const;

  /// @brief Per-loop per-edge connectivity (dead loops filtered out).
  ///
  /// For each live loop id, an inner range yields, per edge of that
  /// loop, the list of neighbouring loop ids that share the edge.
  /// Dead loops have an empty top-level entry; references to dead
  /// loops are removed from the per-edge lists of live loops.
  auto connectivity_per_face_edge() const {
    return tf::make_offset_block_range(
        _loop_edge_offsets,
        tf::make_offset_block_range(_edge_neighbour_offsets,
                                     tf::make_range(_neighbours)));
  }

  /// @brief Per surviving loop, the list of coplanar duplicate loops
  ///        folded into it.
  ///
  /// Sorted by `(survivor_id, dead_id)`. Each entry
  /// `std::array<Index, 3>` is `(survivor, dead, reversed_flag)`.
  /// `reversed_flag == 1` iff the dead loop's winding is reversed
  /// relative to the survivor; `0` iff same direction. Use
  /// `std::lower_bound` on the first column to find the dead-id run
  /// for a given survivor.
  auto coplanar_pairs() const { return tf::make_range(_coplanar_pairs); }

  /// @brief Per-component open/closed flag (`1` = touches a boundary
  ///        edge in the original mesh). Size: `n_components()`.
  auto open_component_mask() const {
    return tf::make_range(_open_component_mask);
  }

  /// @brief Look up the tag a global loop id belongs to.
  ///
  /// O(log n_tags) — uses an upper-bound search on the tag-loop
  /// offset table.
  auto tag_of_loop(Index loop_id) const -> Index {
    auto it = std::upper_bound(_tag_loop_offsets.begin(),
                                _tag_loop_offsets.end(), loop_id);
    return static_cast<Index>((it - _tag_loop_offsets.begin()) - 1);
  }

  /// @brief Reset all stored data.
  auto clear() -> void {
    _polygon_labels.clear();
    _loop_labels.clear();
    _tag_loop_offsets.clear();
    _loop_edge_offsets.clear();
    _edge_neighbour_offsets.clear();
    _neighbours.clear();
    _coplanar_pairs.clear();
    _open_component_mask.clear();
    _n_components = 0;
  }

  /// @brief Populate the graph from an intersection graph, face cuts,
  ///        and the underlying forms.
  ///
  /// Builds the per-tag polygon labels, the flat loop labels, and a
  /// connectivity buffer with coplanar duplicates removed. Performs
  /// no wedge classification — that is the responsibility of
  /// downstream consumers.
  ///
  /// The forms are read for their `manifold_edge_link` policies and
  /// loop geometry only; no reference to them is retained after this
  /// call returns.
  ///
  /// @tparam Int        The exact-arithmetic integer type carried
  ///                    through the intersection pipeline.
  /// @tparam FormsRange Random-access range of @ref tf::polygons
  ///                    forms tagged with at least
  ///                    `manifold_edge_link`.
  /// @param ig    The @ref tf::intersection_graph built for `forms`.
  /// @param fc    The @ref tf::face_cuts derived from `ig`.
  /// @param forms The N tagged polygon forms.
  template <typename Int, typename FormsRange>
  auto build(const tf::intersection_graph<Index, Int> &ig,
             const tf::face_cuts<Index, Int> &fc,
             const FormsRange &forms) -> void {
    clear();

    const auto n_tags = static_cast<Index>(forms.size());
    _polygon_labels.resize(static_cast<std::size_t>(n_tags));
    tf::small_vector<Index, 4> n_surface_per_tag;
    n_surface_per_tag.resize(n_tags);
    Index K_cut = 0;

    tbb::parallel_invoke(
        [&] { _compute_surface_labels(forms, fc, n_surface_per_tag); },
        [&] {
          _compute_loop_connectivity(ig, fc);
          tf::buffer<bool> dead_mask;
          if (_detect_coplanar_dead_loops(fc, dead_mask))
            _clean_connectivity(dead_mask);
          auto cut_cl = _compute_loop_component_labels(fc);
          K_cut = cut_cl.n_components;
          _loop_labels = std::move(cut_cl.labels);
        });

    // ---- Compose global id space. ------------------------------------
    auto form_offsets = _offset_surface_labels(n_surface_per_tag, K_cut);
    const Index total_nodes = form_offsets[form_offsets.size() - 1];

    // ---- Phase 5: collect bridges (surface ↔ cut) and compact. -------
    auto bridges = _collect_bridges(fc, forms);
    _apply_bridges_and_compact(bridges, total_nodes);

    // ---- Phase 6: tag-loop offsets (for tag_of_loop lookup). ---------
    auto tag_offs = fc.tag_offsets();
    _tag_loop_offsets.allocate(tag_offs.size());
    tf::parallel_copy(tag_offs, tf::make_range(_tag_loop_offsets));

    // ---- Phase 7: per-component open/closed flag. --------------------
    _compute_open_components(fc, forms);
  }

private:
  // ====================================================================
  // Private build helpers.
  // ====================================================================

  /// @brief Build per-loop per-edge connectivity from `ig` + `fc`.
  ///
  /// Populates @ref _loop_edge_offsets, @ref _edge_neighbour_offsets,
  /// and @ref _neighbours.
  ///
  /// At this point the connectivity is the *full* graph: every loop
  /// has its edges listed; every edge lists every loop sharing it.
  /// Coplanar dedup happens in a later step that filters this in
  /// place.
  ///
  /// Reuses the same flat-id + face_membership +
  /// `compute_face_link_per_edge` strategy that `cut_graph` does
  /// today, but writes into the graph's own storage so we don't keep
  /// any references to `ig` / `fc` / `forms` after build.
  template <typename Int>
  auto _compute_loop_connectivity(const tf::intersection_graph<Index, Int> &ig,
                                   const tf::face_cuts<Index, Int> &fc) -> void {
    using source = tf::intersect::graph::vertex_source;

    const auto &src = fc.loops_buffer();
    if (!src.size()) {
      _loop_edge_offsets.clear();
      _edge_neighbour_offsets.clear();
      _neighbours.clear();
      return;
    }

    // ---- 1. Per-tag dedup hash maps for original-source vertex ids ----
    auto n_ipts = static_cast<Index>(ig.points().size());
    auto tag_offs = fc.tag_offsets();
    auto n_tags = tag_offs.size() - 1;

    tf::core::std_vector<tf::hash_map<Index, Index>> maps(n_tags);
    auto tag_loops = tf::make_offset_block_range(tag_offs, fc.loops());
    tf::parallel_for_each(tf::enumerate(tag_loops), [&](auto pair) {
      auto &&[tag, loops] = pair;
      auto &m = maps[tag];
      for (auto &&loop : loops)
        for (auto &&v : loop)
          if (v.source == source::original)
            if (m.find(v.id) == m.end())
              m[v.id] = Index(m.size());
    });

    tf::buffer<Index> tag_base;
    tag_base.allocate(n_tags + 1);
    tag_base[0] = n_ipts;
    for (std::size_t t = 0; t < n_tags; ++t)
      tag_base[t + 1] = tag_base[t] + Index(maps[t].size());

    // ---- 2. Build flat per-loop-vertex int ids ----
    tf::buffer<Index> flat_data;
    flat_data.allocate(src.data_buffer().size());
    auto tag_src = tf::make_offset_block_range(
        tag_offs,
        tf::make_offset_block_range(src.offsets_buffer(), src.data_buffer()));
    auto tag_dst = tf::make_offset_block_range(
        tag_offs,
        tf::make_offset_block_range(src.offsets_buffer(), flat_data));
    tf::parallel_for_each(
        tf::enumerate(tf::zip(tag_src, tag_dst)), [&](auto pair) {
          auto &&[tag, tup] = pair;
          auto &&[src_loops, dst_loops] = tup;
          auto base = tag_base[tag];
          auto &m = maps[tag];
          for (auto &&[s, d] : tf::zip(src_loops, dst_loops))
            for (std::size_t i = 0; i < s.size(); ++i)
              d[i] = (s[i].source == source::created) ? s[i].id
                                                       : base + m[s[i].id];
        });

    Index n_flat = tag_base[n_tags];

    // ---- 3. face_membership over flat loops ----
    auto flat_loops = tf::make_offset_block_range(src.offsets_buffer(),
                                                   tf::make_range(flat_data));
    tf::face_membership<Index> fm;
    fm.build(tf::make_faces(flat_loops), n_flat, flat_data.size());

    // ---- 4. Per-edge neighbour lists ----
    tf::topology::compute_face_link_per_edge(
        flat_loops, fm, _edge_neighbour_offsets, _neighbours);

    // ---- 5. Per-loop offsets are exactly the loop-vertex offsets
    //         (each loop has K vertices = K edges).
    _loop_edge_offsets.allocate(src.offsets_buffer().size());
    tf::parallel_copy(tf::make_range(src.offsets_buffer()),
                      tf::make_range(_loop_edge_offsets));
  }

  /// @brief Detect coplanar duplicate loops and mark the losers.
  ///
  /// Two loops that occupy the same plane and overlap as cyclic
  /// vertex sequences are coplanar duplicates. They surface as
  /// edge-0 neighbours of each other for which
  /// `tf::compare_faces(loops[li], loops[nj]) != 0`.
  ///
  /// Each loop self-checks: am I the smallest member of my coplanar
  /// pack under the order `(tag, loop_id)`? If any edge-0 neighbour
  /// is coplanar with me and has a smaller `(tag, loop_id)`, then I
  /// am dead. Otherwise I am the survivor.
  ///
  /// This applies uniformly to cross-tag *and* same-tag coplanar
  /// packs of any size (3+-way packs produce consistent results
  /// because every member detects the same smallest representative).
  /// Each loop writes only its own bit, so no contention.
  ///
  /// Pre-conditions:
  ///   - @ref _compute_loop_connectivity has been run.
  ///
  /// @param fc        The @ref tf::face_cuts the connectivity was
  ///                  derived from.
  /// @param dead_mask Output mask. Always allocated and filled
  ///                  on entry — the bit is set only for losers.
  /// @return `true` iff any loop was marked dead.
  template <typename Int>
  auto _detect_coplanar_dead_loops(const tf::face_cuts<Index, Int> &fc,
                                    tf::buffer<bool> &dead_mask) -> bool {
    auto descs = fc.descriptors();
    auto loops = fc.loops();
    const Index n_loops = static_cast<Index>(loops.size());
    if (n_loops == 0)
      return false;

    // Each pair emitted once from the smaller-(tag, loop_id) side.
    // Vertex equality is (source, id), which is per-form for originals
    // — wrap each vertex with `(effective_tag, v)` so cross-tag
    // comparison disambiguates same-id originals from different forms.
    using vt = tf::intersect::graph::vertex<Index>;
    using key_t = std::pair<Index, vt>;
    auto with_tag = [](const auto &loop, Index tag) {
      return tf::make_mapped_range(loop, [tag](const vt &v) -> key_t {
        const Index eff =
            (v.source == tf::intersect::graph::vertex_source::created)
                ? Index(-1) : tag;
        return key_t{eff, v};
      });
    };

    auto conn = connectivity_per_face_edge();
    tf::generic_generate(
        tf::enumerate(tf::zip(loops, conn)), _coplanar_pairs,
        [&](const auto &item, auto &out) {
          auto &&[li_z, tup] = item;
          auto &&[loop, edge_neighbors] = tup;
          if (edge_neighbors.size() == 0 || edge_neighbors[0].size() == 0)
            return;
          const Index li = static_cast<Index>(li_z);
          const auto my_tag = descs[li].tag;
          for (auto nj : edge_neighbors[0]) {
            const auto nj_tag = descs[nj].tag;
            // Emit once per unordered pair, from the smaller-key side.
            const bool nj_is_smaller =
                (nj_tag < my_tag) || (nj_tag == my_tag && nj < li);
            if (nj_is_smaller)
              continue;
            int cmp = tf::compare_faces(with_tag(loop, my_tag),
                                         with_tag(loops[nj], nj_tag));
            if (cmp == 0)
              continue;
            out.push_back({li, nj, cmp < 0 ? Index(1) : Index(0)});
          }
        });

    // Sort by (survivor, dead) for binary-search lookup later.
    tbb::parallel_sort(_coplanar_pairs.begin(), _coplanar_pairs.end());

    // Fill dead_mask: each pair's [1] = dead.
    dead_mask.allocate(static_cast<std::size_t>(n_loops));
    tf::parallel_fill(dead_mask, false);
    tf::parallel_for_each(_coplanar_pairs, [&](const auto &p) {
      dead_mask[p[1]] = true;
    });

    return _coplanar_pairs.size() > 0;
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
  auto _clean_connectivity(const tf::buffer<bool> &dead_loops_mask) -> void {
    const Index n_loops = static_cast<Index>(_loop_edge_offsets.size() - 1);
    if (n_loops == 0)
      return;

    // ---- 1. Per-loop kept edge count, written directly into the new
    //         offsets buffer at index l+1; then in-place prefix sum.
    tf::buffer<Index> new_loop_edge_offsets;
    new_loop_edge_offsets.allocate(static_cast<std::size_t>(n_loops + 1));
    new_loop_edge_offsets[0] = Index(0);
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      new_loop_edge_offsets[l + 1] = dead_loops_mask[l]
          ? Index(0)
          : Index(_loop_edge_offsets[l + 1] - _loop_edge_offsets[l]);
    });
    std::partial_sum(new_loop_edge_offsets.begin(),
                     new_loop_edge_offsets.end(),
                     new_loop_edge_offsets.begin());
    const Index new_n_edges = new_loop_edge_offsets[n_loops];

    // ---- 2. Per kept edge: count live neighbours directly into the
    //         new edge-offsets buffer at index e+1; then in-place
    //         prefix sum.
    tf::buffer<Index> new_edge_neighbour_offsets;
    new_edge_neighbour_offsets.allocate(static_cast<std::size_t>(new_n_edges + 1));
    new_edge_neighbour_offsets[0] = Index(0);
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      if (dead_loops_mask[l]) return;
      const Index old_first = _loop_edge_offsets[l];
      const Index old_last  = _loop_edge_offsets[l + 1];
      Index new_edge_idx = new_loop_edge_offsets[l];
      for (Index e = old_first; e < old_last; ++e, ++new_edge_idx) {
        const Index nb_first = _edge_neighbour_offsets[e];
        const Index nb_last  = _edge_neighbour_offsets[e + 1];
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
      if (dead_loops_mask[l]) return;
      const Index old_first = _loop_edge_offsets[l];
      const Index old_last  = _loop_edge_offsets[l + 1];
      Index new_edge_idx = new_loop_edge_offsets[l];
      for (Index e = old_first; e < old_last; ++e, ++new_edge_idx) {
        const Index nb_first = _edge_neighbour_offsets[e];
        const Index nb_last  = _edge_neighbour_offsets[e + 1];
        Index write_pos = new_edge_neighbour_offsets[new_edge_idx];
        for (Index k = nb_first; k < nb_last; ++k) {
          const Index nb = _neighbours[k];
          if (!dead_loops_mask[nb])
            new_neighbours[write_pos++] = nb;
        }
      }
    });

    _loop_edge_offsets      = std::move(new_loop_edge_offsets);
    _edge_neighbour_offsets = std::move(new_edge_neighbour_offsets);
    _neighbours             = std::move(new_neighbours);
  }

  /// @brief Whole-pool connected-component labelling over cut loops.
  ///
  /// One CCL over every cut loop across every form, with the **only**
  /// no-cross rule being non-manifold edges (edges with ≠ 1 neighbour
  /// in the cleaned connectivity). Components naturally span tags
  /// wherever the local connectivity is manifold — which matches the
  /// semantics of materialised-MEL CCL on the merged arrangement
  /// mesh.
  ///
  /// Dead loops (coplanar duplicates dropped by
  /// @ref _clean_connectivity) have empty per-loop edge ranges. We
  /// mask them off so the CCL leaves their label at `none_label`,
  /// mirroring how @ref make_surface_component_labels2 masks off cut
  /// faces.
  ///
  /// Pre-conditions:
  ///   - @ref _compute_loop_connectivity has been run.
  ///   - @ref _clean_connectivity has been run iff any coplanar
  ///     duplicates were detected.
  ///
  /// Returns a single @ref tf::connected_component_labels whose
  /// `labels[loop_id]` is either a component id in
  /// `[0, n_components)` or `none_label` for dropped loops.
  template <typename Int>
  auto _compute_loop_component_labels(const tf::face_cuts<Index, Int> &fc) const
      -> tf::connected_component_labels<Index> {
    const auto n_loops = static_cast<Index>(fc.loops().size());

    tf::connected_component_labels<Index> result;
    result.labels.allocate(static_cast<std::size_t>(n_loops));

    // Build the per-loop alive mask. Loops with empty edge ranges
    // (dead via coplanar dedup) are masked off and pre-seeded with
    // none_label so the CCL leaves them untouched.
    tf::buffer<char> mask;
    mask.allocate(static_cast<std::size_t>(n_loops));
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      const bool alive =
          _loop_edge_offsets[l + 1] > _loop_edge_offsets[l];
      mask[l] = alive ? char(1) : char(0);
      if (!alive) result.labels[l] = none_label;
    });

    // Cross only manifold edges within a single tag. Cross-tag
    // manifold neighbours (two forms gluing on an edge) are distinct
    // surfaces — components must stay per-tag for downstream code
    // (volumes, B(c), bundle_to_tags) to behave correctly.
    auto descs = fc.descriptors();
    auto applier = [this, descs](Index loop_id, const auto &f) {
      const Index my_tag = descs[loop_id].tag;
      const Index edge_first = _loop_edge_offsets[loop_id];
      const Index edge_last  = _loop_edge_offsets[loop_id + 1];
      for (Index e = edge_first; e < edge_last; ++e) {
        const Index nb_first = _edge_neighbour_offsets[e];
        const Index nb_last  = _edge_neighbour_offsets[e + 1];
        if (nb_last - nb_first != Index(1)) continue;   // non-manifold
        const Index nb = _neighbours[nb_first];
        if (descs[nb].tag != my_tag) continue;          // cross-tag
        f(nb);
      }
    };

    result.n_components = tf::label_connected_components_masked(
        result.labels, mask, applier, Index(4));
    return result;
  }

  /// @brief Build initial per-form surface labels + whole-pool cut
  ///        labels, all expressed in a single global id space.
  ///
  /// Layout of the global id space after this method returns:
  ///   - `[0, K_cut)`              — cut-loop ids in @ref _loop_labels.
  ///   - `[K_cut + Σ_{s<t} K_s, K_cut + Σ_{s<=t} K_s)` — surface ids
  ///     for form `t` in @ref _polygon_labels[t].
  ///
  /// The id space is **not yet compacted**: bridges that union a
  /// surface component with a cut component still need to be
  /// collected, fed into a dense equivalence-class map, and applied.
  /// That is the responsibility of the next two methods in the
  /// pipeline (`_collect_bridges` and `_apply_equivalence_map`).
  ///
  /// Pre-conditions:
  ///   - @ref _compute_loop_connectivity and (if any coplanar
  ///     duplicates were found) @ref _clean_connectivity have run.
  ///   - Each `forms[t]` must carry a `manifold_edge_link` tag.
  ///
  /// @param fc     The @ref tf::face_cuts derived from the arrangement.
  /// @param forms  The N tagged polygon forms.
  /// @return Per-form prefix-sum offsets into the global id space.
  ///         `form_offsets[t]`     = first id of form `t`'s surface
  ///         component slot. `form_offsets[n_tags]` = total node count
  ///         (the size to allocate for the equivalence-class map).
  /// @brief Per-form surface CCL, in parallel.
  ///
  /// Independent of the cut connectivity — only reads `forms` (each
  /// form's `manifold_edge_link`) and `fc.descriptors`/`fc.tag_offsets`.
  /// Writes per-tag labels into `_polygon_labels[t]` and the per-tag
  /// component count into `n_surface_per_tag[t]`. Labels are still in
  /// each form's local id space — @ref _offset_surface_labels lifts
  /// them into the global space once `K_cut` is known.
  template <typename Int, typename FormsRange>
  auto _compute_surface_labels(
      const FormsRange &forms, const tf::face_cuts<Index, Int> &fc,
      tf::small_vector<Index, 4> &n_surface_per_tag) -> void {
    const auto n_tags = static_cast<Index>(forms.size());
    auto descs_per_tag =
        tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());

    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t) {
      tg.run([this, t, &forms, &descs_per_tag, &n_surface_per_tag] {
        auto cl = tf::cut::make_surface_component_labels2<Index, label_type>(
            forms[t], descs_per_tag[t]);
        n_surface_per_tag[t] = cl.n_components;
        _polygon_labels[static_cast<std::size_t>(t)] = std::move(cl.labels);
      });
    }
    tg.wait();
  }

  /// @brief Lift per-form surface labels into the global id space.
  ///
  /// Returns `form_offsets` with `form_offsets[t]` = first id of form
  /// `t`'s surface slot and `form_offsets[n_tags]` = total node count.
  /// Mutates `_polygon_labels[t]` in place to add the offset.
  auto _offset_surface_labels(
      const tf::small_vector<Index, 4> &n_surface_per_tag, Index K_cut)
      -> tf::buffer<Index> {
    const auto n_tags = static_cast<Index>(n_surface_per_tag.size());
    tf::buffer<Index> form_offsets;
    form_offsets.allocate(static_cast<std::size_t>(n_tags + 1));
    form_offsets[0] = K_cut;
    for (Index t = 0; t < n_tags; ++t)
      form_offsets[t + 1] = form_offsets[t] + n_surface_per_tag[t];

    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t)
      tg.run([this, t, &form_offsets] {
        const Index offset = form_offsets[t];
        tf::parallel_for_each(
            _polygon_labels[static_cast<std::size_t>(t)],
            [offset](auto &id) {
              if (id != none_label) id += offset;
            },
            tf::checked);
      });
    tg.wait();
    return form_offsets;
  }

  /// @brief Walk live loops and emit `(min, max)` bridge pairs between
  ///        a cut-loop component and the surface component of an
  ///        uncut neighbour face that touches it across the
  ///        source mesh's manifold edge.
  ///
  /// Mirrors the inner generic_generate body of
  /// `tf::cut::merge_labels`: an edge of a live loop with **zero**
  /// neighbours in the cleaned cut connectivity (so not an
  /// intersection edge) AND whose loop vertex sub-id lives on an
  /// original-mesh vertex or edge IS a candidate bridge. We then ask
  /// the source mesh's MEL for the peer face. If that peer is uncut
  /// (its global surface label != `none_label`) we have a bridge.
  ///
  /// Pre-conditions:
  ///   - @ref _build_initial_labels has run.
  ///   - `_polygon_labels[t]` and `_loop_labels` are populated in the
  ///     global id space (not yet compacted).
  ///   - Each `forms[t]` carries a `manifold_edge_link` tag.
  ///
  /// @return A deduplicated buffer of `(min, max)` pairs of global ids.
  template <typename Int, typename FormsRange>
  auto _collect_bridges(const tf::face_cuts<Index, Int> &fc,
                        const FormsRange &forms) const
      -> tf::buffer<std::array<Index, 2>> {
    tf::buffer<std::array<Index, 2>> bridges;

    auto conn = connectivity_per_face_edge();

    tf::generic_generate(
        tf::zip(tf::make_range(_loop_labels),
                fc.descriptors(), conn, fc.loops()),
        bridges,
        tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>>{},
        [&](auto tup, auto &buffer, auto &set) {
          auto &&[cut_id, d, edge_conn, loop] = tup;
          if (cut_id == none_label) return;
          const auto &poly_labels_t =
              _polygon_labels[static_cast<std::size_t>(d.tag)];
          auto mel_t = forms[d.tag].manifold_edge_link();
          const auto face_size = forms[d.tag].faces()[d.object].size();
          const auto n = loop.size();
          for (std::size_t i = 0; i < n; ++i) {
            // Only candidate-bridge loop edges with no in-arrangement
            // neighbour (interior cuts already paired via connectivity).
            if (edge_conn[i].size() != 0) continue;
            const auto next = tf::circular_increment(i, n);
            const auto eidx = tf::cut::resolve_face_edge(
                loop[i].sub_id, loop[next].sub_id, face_size);
            if (!eidx) continue;
            auto m = mel_t[d.object][*eidx];
            if (!m.is_simple()) continue;
            const Index peer = m.face_peer;
            const Index peer_label = poly_labels_t[peer];
            if (peer_label == none_label) continue;
            const Index a = std::min(peer_label, Index(cut_id));
            const Index b = std::max(peer_label, Index(cut_id));
            std::array<Index, 2> pair{a, b};
            if (set.insert(pair).second)
              buffer.push_back(pair);
          }
        });

    return bridges;
  }

  /// @brief Mark every component touching a boundary edge as open.
  ///
  /// Two parallel passes (one task per form + one task for the loop
  /// stream, all racing in a `tbb::task_group`):
  ///   - Live loops: any vertex whose `sub_id` lies on an original
  ///     face vertex/edge, with `mel.is_boundary()` true at that
  ///     edge, flags the loop's component.
  ///   - Uncut faces: any boundary edge in the form's MEL flags the
  ///     face's component.
  ///
  /// `connectivity_per_face_edge()` neighbour counts are NOT used —
  /// they don't distinguish mesh boundaries from bridges or cross-tag
  /// coincidences. Writes are `1`-only, idempotent under data race.
  template <typename Int, typename FormsRange>
  auto _compute_open_components(const tf::face_cuts<Index, Int> &fc,
                                 const FormsRange &forms) -> void {
    _open_component_mask.allocate(static_cast<std::size_t>(_n_components));
    tf::parallel_fill(_open_component_mask, char(0));

    if (_n_components == 0)
      return;

    const Index n_tags = static_cast<Index>(_polygon_labels.size());
    tbb::task_group tg;

    // Pass A: live loops.
    tg.run([this, &fc, &forms] {
      tf::parallel_for_each(
          tf::zip(tf::make_range(_loop_labels), fc.descriptors(), fc.loops()),
          [this, &forms](auto tup) {
            auto &&[cut_id, d, loop] = tup;
            if (cut_id == none_label)
              return;
            auto mel_t = forms[d.tag].manifold_edge_link();
            const auto face_size = forms[d.tag].faces()[d.object].size();
            const auto n = loop.size();
            for (std::size_t i = 0; i < n; ++i) {
              const auto next = tf::circular_increment(i, n);
              const auto eidx = tf::cut::resolve_face_edge(
                  loop[i].sub_id, loop[next].sub_id, face_size);
              if (!eidx)
                continue;
              auto m = mel_t[d.object][*eidx];
              if (m.is_boundary())
                _open_component_mask[static_cast<std::size_t>(cut_id)] =
                    char(1);
            }
          });
    });

    // Pass B: one task per form, each parallel over its uncut faces.
    for (Index t = Index(0); t < n_tags; ++t) {
      tg.run([this, t, &forms] {
        auto poly_labels_t = tf::make_range(_polygon_labels[t]);
        auto mel_t = forms[t].manifold_edge_link();
        tf::parallel_for_each(
            tf::enumerate(poly_labels_t), [this, mel_t](auto pair) {
              auto &&[f, label] = pair;
              if (label == none_label)
                return;
              auto face_mel = mel_t[Index(f)];
              for (const auto &m : face_mel) {
                if (m.is_boundary()) {
                  _open_component_mask[static_cast<std::size_t>(label)] =
                      char(1);
                  return;
                }
              }
            });
      });
    }
    tg.wait();
  }

  /// @brief Finalise the global id space: collapse bridge pairs into
  ///        connected components and apply the remap to every label
  ///        buffer.
  ///
  /// Pre-conditions:
  ///   - @ref _build_initial_labels and @ref _collect_bridges have run.
  ///   - `_polygon_labels[t]` and `_loop_labels` are in the
  ///     pre-compaction global id space, `none_label` for cut faces
  ///     and dropped loops respectively.
  ///
  /// On return, both label buffers carry the final, dense, contiguous
  /// component ids in `[0, n_components())`. `_n_components` is set.
  ///
  /// Mirrors the final-stage pattern of `tf::cut::merge_labels`:
  ///   1. `tbb::parallel_sort` + `std::unique` global dedup of pairs.
  ///   2. `tf::make_dense_equivalence_class_map` over the pairs.
  ///   3. In-place `tf::parallel_for_each` remap on every label
  ///      buffer, skipping `none_label` entries.
  ///
  /// @param bridges     The pair buffer returned by @ref _collect_bridges
  ///                    (consumed — sorted + uniqued in place).
  /// @param total_nodes The size of the pre-compaction id space, i.e.
  ///                    `form_offsets[n_tags]` from
  ///                    @ref _build_initial_labels.
  auto _apply_bridges_and_compact(
      tf::buffer<std::array<Index, 2>> &bridges, Index total_nodes) -> void {
    // Global dedup of bridges across thread-local hash-set batches.
    tbb::parallel_sort(bridges.begin(), bridges.end());
    bridges.erase_till_end(std::unique(bridges.begin(), bridges.end()));

    // Dense union-find over [0, total_nodes), one slot per pre-compaction id.
    tf::buffer<Index> map;
    map.allocate(static_cast<std::size_t>(total_nodes));
    _n_components = tf::make_dense_equivalence_class_map(bridges, map);

    // In-place remap of every label buffer; none_label stays.
    auto remap = [&map](auto &id) {
      if (id != none_label) id = map[id];
    };

    tbb::task_group tg;
    tg.run([&] {
      tf::parallel_for_each(_loop_labels, remap, tf::checked);
    });
    for (std::size_t t = 0; t < _polygon_labels.size(); ++t)
      tg.run([&, t] {
        tf::parallel_for_each(_polygon_labels[t], remap, tf::checked);
      });
    tg.wait();
  }

  // ====================================================================
  // Private data.
  // ====================================================================

  // --------------------------------------------------------------------
  // Per-tag surface component labels.
  //   _polygon_labels[t][face_id] = global component id, or none_label
  //   if face_id was cut by the arrangement.
  // --------------------------------------------------------------------
  tf::small_vector<tf::buffer<label_type>, 4> _polygon_labels;

  // --------------------------------------------------------------------
  // Flat per-loop labels across all tags.
  //   _loop_labels[loop_id] = global component id, or none_label if
  //   the loop was dropped as a coplanar duplicate.
  // loop_id ordering matches face_cuts.descriptors().
  // --------------------------------------------------------------------
  tf::buffer<label_type> _loop_labels;

  // --------------------------------------------------------------------
  // Per-tag offsets into _loop_labels: tag t's loops occupy
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

  // --------------------------------------------------------------------
  // Sorted list of coplanar (survivor, dead, reversed) triples.
  //
  // Each entry is std::array<Index, 3>:
  //   [0] = survivor loop id   (lex-min by (tag, loop_id) of the pair)
  //   [1] = dead loop id       (lex-max by (tag, loop_id) of the pair)
  //   [2] = reversed flag      (1 = dead's winding reversed relative to
  //                             survivor; 0 = same direction)
  //
  // Sorted by [0] then [1] for binary-search lookup of "duplicates of
  // survivor id S" via std::lower_bound on the first column.
  //
  // Coplanar packs of size > 2 may contribute pairs where the survivor
  // field is itself a dead loop (e.g., for pack {A, B, C} with A < B <
  // C, we emit (A,B), (A,C), (B,C)). The (B,C) entry is "noise" — B is
  // dead. Downstream queries only consult alive loops, so the noise is
  // harmless. dead_mask is built from the second column of every pair.
  // --------------------------------------------------------------------
  tf::buffer<std::array<Index, 3>> _coplanar_pairs;

  // Per-component open flag (`1` = touches a boundary edge in the
  // input). Filled by `_compute_open_components` at end of `build()`.
  tf::buffer<char> _open_component_mask;

  // --------------------------------------------------------------------
  // Number of distinct arrangement components after compaction.
  // --------------------------------------------------------------------
  Index _n_components = 0;
};

} // namespace tf
