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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/connected_component_labels.hpp"
#include "../../topology/label_connected_components.hpp"
#include "../face_regions.hpp"
#include "../loop_connectivity.hpp"
#include "../partition/make_labels.hpp"
#include "../resolve_face_edge.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

#include <algorithm>
#include <array>

namespace tf::cut {

/// @ingroup cut
/// @brief The classification label tier over a
///        @ref tf::loop_connectivity: whole-pool cut-loop component
///        labels, per-form surface labels, surface↔cut bridges
///        compacted into one dense component id space, and the
///        per-component open (boundary-touching) mask.
///
/// Built by @ref tf::csg_graph — the arrangement tier never reads
/// these. Owns the loop connectivity (built here, cleaned with the
/// arrangement's dead mask — stacks are never re-detected) and
/// delegates its accessors so classification consumers take one
/// object.
///

template <typename Index> class component_labels {
public:
  using label_type = Index;
  static constexpr label_type none_label = label_type(-1);

  auto n_components() const -> Index { return _n_components; }

  /// @brief Per-face surface component labels for form `tag`;
  ///        `none_label` for faces cut by the arrangement.
  auto polygon_labels(Index tag) const {
    return tf::make_range(_polygon_labels[static_cast<std::size_t>(tag)]);
  }

  /// @brief Flat per-loop component labels; dead (coplanar-duplicate)
  ///        loops carry `none_label`.
  auto loop_labels() const { return tf::make_range(_loop_labels); }
  /// Derived per-exposed-triangle labels — each region's label
  /// scattered over its exposed triangle block via @ref bind_exposed.
  /// Classification truth lives at region grain; geometry consumers
  /// iterate triangles.
  auto triangle_labels() const { return tf::make_range(_triangle_labels); }
  /// Materialize the triangle-grain label view from the exposure's
  /// forward map (region -> its exposed triangle block). Blocks are
  /// disjoint by construction; triangles outside every region block
  /// (late-promoted faces) stay at `none_label`.
  template <typename RangesRange>
  auto bind_exposed(const RangesRange &exposed_ranges, std::size_t n_triangles)
      -> void {
    _triangle_labels.allocate(n_triangles);
    tf::parallel_fill(_triangle_labels, none_label);
    const std::size_t n_regions = _loop_labels.size();
    tf::parallel_for_each(tf::make_sequence_range(n_regions),
                          [&](std::size_t li) {
                            const auto r = exposed_ranges[li];
                            const auto label = _loop_labels[li];
                            for (auto t = r[0]; t < r[1]; ++t)
                              _triangle_labels[std::size_t(t)] = label;
                          });
  }

  /// @brief Per-component open flag (`1` = touches a boundary edge).
  auto open_component_mask() const {
    return tf::make_range(_open_component_mask);
  }

  auto n_tags() const -> Index { return _conn.n_tags(); }

  auto connectivity_per_face_edge() const {
    return _conn.connectivity_per_face_edge();
  }
  auto coplanar_pairs() const { return tf::make_range(_coplanar_pairs); }
  auto connectivity() const -> const tf::loop_connectivity<Index> & {
    return _conn;
  }

  auto clear() -> void {
    _conn.clear();
    _polygon_labels.clear();
    _loop_labels.clear();
    _triangle_labels.clear();
    _open_component_mask.clear();
    _n_components = 0;
    _coplanar_pairs.clear();
  }

  /// @brief Label the arrangement's components over the cleaned
  ///        connectivity.
  ///
  /// Surface CCL per form and cut-loop CCL run in parallel; bridges
  /// (surface ↔ cut across the source mesh's manifold edges) union the
  /// two id spaces, compacted to dense component ids.
  ///
  /// @param rt        The arrangement's exposed triangle-grain loops
  ///                   (@ref tf::face_regions, region grain).
  /// @param forms     The N tagged polygon forms (each carrying a
  ///                  `manifold_edge_link`); not retained.
  /// @param pairs     The arrangement's coplanar stack triples at the
  ///                  exposed grain (copied — stacks are few).
  /// @param dead      The arrangement's per-loop dead mask.
  /// @param n_created Size of the graph's unified created-points table.
  template <typename Int, typename FormsRange, typename Pairs,
            typename DeadRange>
  auto build(const tf::face_regions<Index, Int> &fr,
             const FormsRange &forms, const Pairs &pairs, const DeadRange &dead,
             Index n_created) -> void {
    clear();
    tf::buffer<Index> point_counts;
    point_counts.allocate(std::size_t(forms.size()));
    for (std::size_t t = 0; t < std::size_t(forms.size()); ++t)
      point_counts[t] = static_cast<Index>(forms[t].points().size());
    _conn.build(fr, dead, tf::make_range(point_counts), n_created);
    _coplanar_pairs.allocate(std::size_t(pairs.size()));
    for (std::size_t i = 0; i < std::size_t(pairs.size()); ++i)
      _coplanar_pairs[i] = pairs[i];

    const auto n_tags = static_cast<Index>(forms.size());
    _polygon_labels.resize(static_cast<std::size_t>(n_tags));
    tf::small_vector<Index, 4> n_surface_per_tag;
    n_surface_per_tag.resize(n_tags);
    Index K_cut = 0;

    tbb::parallel_invoke(
        [&] { _compute_surface_labels(forms, fr, n_surface_per_tag); },
        [&] {
          auto cut_cl = _compute_loop_component_labels(fr);
          K_cut = cut_cl.n_components;
          _loop_labels = std::move(cut_cl.labels);
        });

    auto form_offsets = _offset_surface_labels(n_surface_per_tag, K_cut);
    const Index total_nodes = form_offsets[form_offsets.size() - 1];

    auto bridges = _collect_bridges(fr, forms);
    _apply_bridges_and_compact(bridges, total_nodes);

    _compute_open_components(fr, forms);
  }

private:
  /// Whole-pool CCL over regions: cross only single-neighbour
  /// (manifold) walk edges within one tag. Dead regions (empty edge
  /// ranges) are masked off at `none_label`. Regions are atomic — a
  /// vertex-pinched region is one node, and interior triangulation
  /// edges do not exist at this grain, so a coincident CDT diagonal
  /// can never fragment or fuse components.
  template <typename Int>
  auto _compute_loop_component_labels(
      const tf::face_regions<Index, Int> &fr) const
      -> tf::connected_component_labels<Index> {
    const auto n_loops = static_cast<Index>(fr.loops().size());
    auto conn = _conn.connectivity_per_face_edge();

    tf::connected_component_labels<Index> result;
    result.labels.allocate(static_cast<std::size_t>(n_loops));

    tf::buffer<char> mask;
    mask.allocate(static_cast<std::size_t>(n_loops));
    tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
      const bool alive = conn[l].size() > 0;
      mask[l] = alive ? char(1) : char(0);
      if (!alive)
        result.labels[l] = none_label;
    });

    auto descs = fr.descriptors();
    auto applier = [conn, descs](Index loop_id, const auto &f) {
      for (auto &&nbrs : conn[loop_id]) {
        if (nbrs.size() != std::size_t(1))
          continue; // non-manifold
        const Index nb = nbrs[0];
        if (descs[nb].tag != descs[loop_id].tag)
          continue; // cross-tag
        f(nb);
      }
    };

    result.n_components = tf::label_connected_components_masked(
        result.labels, mask, applier, Index(4));
    return result;
  }

  /// Per-form surface CCL, in parallel; labels stay in each form's
  /// local id space until @ref _offset_surface_labels lifts them.
  template <typename Int, typename FormsRange>
  auto _compute_surface_labels(const FormsRange &forms,
                               const tf::face_regions<Index, Int> &fr,
                               tf::small_vector<Index, 4> &n_surface_per_tag)
      -> void {
    const auto n_tags = static_cast<Index>(forms.size());
    auto descs_per_tag =
        tf::make_offset_block_range(fr.tag_offsets(), fr.descriptors());

    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t) {
      tg.run([this, t, &forms, &descs_per_tag, &fr, &n_surface_per_tag] {
        auto cl = tf::cut::make_surface_component_labels2<Index, label_type>(
            forms[t], descs_per_tag[t], fr.deleted(t));
        n_surface_per_tag[t] = cl.n_components;
        _polygon_labels[static_cast<std::size_t>(t)] = std::move(cl.labels);
      });
    }
    tg.wait();
  }

  /// Lift per-form surface labels into the global id space; returns
  /// per-form offsets, `form_offsets[n_tags]` = total node count.
  auto
  _offset_surface_labels(const tf::small_vector<Index, 4> &n_surface_per_tag,
                         Index K_cut) -> tf::buffer<Index> {
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
              if (id != none_label)
                id += offset;
            },
            tf::checked);
      });
    tg.wait();
    return form_offsets;
  }

  /// Bridge pairs between a cut-loop component and the surface
  /// component of an uncut neighbour across the source mesh's manifold
  /// edge. Candidate edges have zero in-arrangement neighbours.
  template <typename Int, typename FormsRange>
  auto _collect_bridges(const tf::face_regions<Index, Int> &fr,
                        const FormsRange &forms) const
      -> tf::buffer<std::array<Index, 2>> {
    tf::buffer<std::array<Index, 2>> bridges;

    auto conn = _conn.connectivity_per_face_edge();

    tf::generic_generate(
        tf::zip(tf::make_range(_loop_labels), fr.descriptors(), conn,
                fr.loops()),
        bridges, tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>>{},
        [&](auto tup, auto &buffer, auto &set) {
          auto &&[cut_id, d, edge_conn, loop] = tup;
          if (cut_id == none_label)
            return;
          const auto &poly_labels_t =
              _polygon_labels[static_cast<std::size_t>(d.tag)];
          auto mel_t = forms[d.tag].manifold_edge_link();
          const auto face_size = forms[d.tag].faces()[d.object].size();
          const auto n = loop.size();
          for (std::size_t i = 0; i < n; ++i) {
            if (edge_conn[i].size() != 0)
              continue;
            const auto next = tf::circular_increment(i, n);
            const auto eidx = tf::cut::resolve_face_edge(
                loop[i].sub_id, loop[next].sub_id, face_size);
            if (!eidx)
              continue;
            auto m = mel_t[d.object][*eidx];
            if (!m.is_simple())
              continue;
            const Index peer = m.face_peer;
            const Index peer_label = poly_labels_t[peer];
            if (peer_label == none_label)
              continue;
            const Index a = std::min(peer_label, Index(cut_id));
            const Index b = std::max(peer_label, Index(cut_id));
            std::array<Index, 2> pair{a, b};
            if (set.insert(pair).second)
              buffer.push_back(pair);
          }
        });

    return bridges;
  }

  /// Collapse bridge pairs into components and remap every label
  /// buffer to the final dense ids.
  auto _apply_bridges_and_compact(tf::buffer<std::array<Index, 2>> &bridges,
                                  Index total_nodes) -> void {
    tbb::parallel_sort(bridges.begin(), bridges.end());
    bridges.erase_till_end(std::unique(bridges.begin(), bridges.end()));

    tf::buffer<Index> map;
    map.allocate(static_cast<std::size_t>(total_nodes));
    _n_components = tf::make_dense_equivalence_class_map(bridges, map);

    auto remap = [&map](auto &id) {
      if (id != none_label)
        id = map[id];
    };

    tbb::task_group tg;
    tg.run([&] { tf::parallel_for_each(_loop_labels, remap, tf::checked); });
    for (std::size_t t = 0; t < _polygon_labels.size(); ++t)
      tg.run([&, t] {
        tf::parallel_for_each(_polygon_labels[t], remap, tf::checked);
      });
    tg.wait();
  }

  /// Mark every component touching a boundary edge as open. Writes are
  /// `1`-only, idempotent under data race.
  template <typename Int, typename FormsRange>
  auto _compute_open_components(const tf::face_regions<Index, Int> &fr,
                                const FormsRange &forms) -> void {
    _open_component_mask.allocate(static_cast<std::size_t>(_n_components));
    tf::parallel_fill(_open_component_mask, char(0));

    if (_n_components == 0)
      return;

    const Index n_tags = static_cast<Index>(_polygon_labels.size());
    tbb::task_group tg;

    tg.run([this, &fr, &forms] {
      auto holes = fr.holes();
      auto loop_holes = fr.loop_holes();
      tf::parallel_for_each(
          tf::zip(tf::make_sequence_range(std::size_t(fr.loops().size())),
                  tf::make_range(_loop_labels), fr.descriptors(), fr.loops()),
          [this, &forms, &holes, &loop_holes](auto tup) {
            auto &&[li_b, cut_id_b, d_b, loop_b] = tup;
            const auto li = li_b;
            const auto cut_id = cut_id_b;
            const auto d = d_b;
            const auto &loop = loop_b;
            if (cut_id == none_label)
              return;
            auto mel_t = forms[d.tag].manifold_edge_link();
            const auto face_size = forms[d.tag].faces()[d.object].size();
            auto scan = [&](const auto &walk) {
              const auto n = walk.size();
              for (std::size_t i = 0; i < n; ++i) {
                const auto next = tf::circular_increment(i, n);
                const auto eidx = tf::cut::resolve_face_edge(
                    walk[i].sub_id, walk[next].sub_id, face_size);
                if (!eidx)
                  continue;
                auto m = mel_t[d.object][*eidx];
                if (m.is_boundary())
                  _open_component_mask[static_cast<std::size_t>(cut_id)] =
                      char(1);
              }
            };
            scan(loop);
            for (auto h : loop_holes[Index(li)])
              scan(holes[h]);
          });
    });

    for (Index t = Index(0); t < n_tags; ++t) {
      tg.run([this, t, &forms] {
        auto poly_labels_t = tf::make_range(_polygon_labels[t]);
        auto mel_t = forms[t].manifold_edge_link();
        tf::parallel_for_each(tf::enumerate(poly_labels_t), [this,
                                                             mel_t](auto pair) {
          auto &&[f, label] = pair;
          if (label == none_label)
            return;
          auto face_mel = mel_t[Index(f)];
          for (const auto &m : face_mel) {
            if (m.is_boundary()) {
              _open_component_mask[static_cast<std::size_t>(label)] = char(1);
              return;
            }
          }
        });
      });
    }
    tg.wait();
  }

  tf::loop_connectivity<Index> _conn;
  tf::small_vector<tf::buffer<label_type>, 4> _polygon_labels;
  tf::buffer<label_type> _loop_labels;
  tf::buffer<label_type> _triangle_labels;
  tf::buffer<char> _open_component_mask;
  Index _n_components = 0;
  tf::buffer<std::array<Index, 3>> _coplanar_pairs;
};

} // namespace tf::cut
