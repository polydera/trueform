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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/edges.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/edge_path_connector.hpp"
#include "../arrangement_graph.hpp"
#include "../face_cuts.hpp"
#include "./arrangement_descriptor.hpp"
#include "./build_domain_of_side.hpp"
#include "./canonicalize_nm_edges.hpp"
#include "./compute_bundle_tag_index.hpp"
#include "./compute_majority_rep.hpp"
#include "./emit_domain_merges.hpp"
#include "./make_bundle_labels.hpp"
#include "./make_non_manifold_edge_fans.hpp"
#include <array>
#include <utility>

namespace tf::cut {

/// @ingroup cut
/// @brief Hasher for `intersect::graph::vertex<Index>` over `(source,
///        id)` — `sub_id` is ignored to match `operator==`.
template <typename Index> struct arrangement_vertex_hash {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  auto operator()(const vertex_t &v) const -> std::size_t {
    return tf::array_hash<Index, 2>{}(
        std::array<Index, 2>{Index(v.source), v.id});
  }
};

/// @ingroup cut
/// @brief Build the @ref tf::cut::arrangement_descriptor for an
///        implicit N-form arrangement.
///
/// Orchestrates the wedge-merging half of the domain pipeline:
///
///   1. Enumerate non-manifold edge fans from the graph's connectivity.
///   2. Group NM edges into polylines and align per-edge direction
///      with the polyline orientation via
///      @ref tf::edge_path_connector (keyed on `vertex_t`).
///   3. CCW-radial-sort each fan and write set-canonical keys
///      (@ref tf::cut::canonicalize_nm_edges).
///   4. Pick one representative NM edge per `(path_id, set-key)`
///      bucket (@ref tf::cut::compute_majority_rep).
///   5. Walk reps cyclically and emit component-side and
///      component-only merge pairs (@ref tf::cut::emit_domain_merges).
///   6. Dense union-find to obtain `domain_of_side` and
///      `bundle_of_component`.
///
/// Nesting (domain–domain containment of disjoint shells) and
/// per-form sidedness classification are downstream stages that build
/// on this descriptor; this function does not perform them.
///
/// @tparam Int      Exact integer type for predicate intermediates in
///                  @ref tf::cut::canonicalize_nm_edges.
/// @tparam GetPoint Callable `(vertex_t v, Index tag) ->
///                  tf::point<Int, 3>`. Provides per-tag context for
///                  resolving original-source vertex positions.
///
/// @param is_sheet_tag Per-form sheet mask (may be empty: no sheets).
///        Open components of sheet forms are exempt from the Mode-2
///        self-merge: a sheet's fragments always separate their two
///        sides, realising the virtual extension of an open cutter.
template <typename Int, typename Index, typename Index1, typename GetPoint,
          typename ApplyToFace>
auto make_arrangement_descriptor(const tf::arrangement_graph<Index> &ag,
                                 const tf::face_cuts<Index, Index1> &fc,
                                 GetPoint get_point, ApplyToFace apply_to_face,
                                 const tf::buffer<char> &is_sheet_tag = {})
    -> arrangement_descriptor<Index> {
  using vertex_t = typename tf::cut::non_manifold_edge_fans<Index>::vertex_t;
  using ag_t = tf::arrangement_graph<Index>;

  arrangement_descriptor<Index> out;
  const Index n_components = ag.n_components();

  // Components of sheet forms never self-merge; mark them before
  // compute_bundle_tag_index runs. Only sheet geometry is touched.
  tf::buffer<char> sheet_component;
  if (is_sheet_tag.size() > 0) {
    sheet_component.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_fill(sheet_component, char(0));
    auto loop_labels = ag.loop_labels();
    auto descs = fc.descriptors();
    tf::parallel_for_each(
        tf::make_sequence_range(static_cast<Index>(loop_labels.size())),
        [&](Index l) {
          const Index t = descs[l].tag;
          if (t < Index(0) || t >= Index(is_sheet_tag.size()) ||
              !is_sheet_tag[t])
            return;
          const Index c = loop_labels[l];
          if (c != ag_t::none_label)
            sheet_component[c] = char(1);
        });
    const Index n_tags = ag.n_tags();
    for (Index t = Index(0); t < n_tags; ++t) {
      if (t >= Index(is_sheet_tag.size()) || !is_sheet_tag[t])
        continue;
      auto labels = ag.polygon_labels(t);
      tf::parallel_for_each(
          tf::make_sequence_range(static_cast<Index>(labels.size())),
          [&](Index f) {
            const Index c = labels[f];
            if (c != ag_t::none_label)
              sheet_component[c] = char(1);
          });
    }
  }
  auto is_sheet_component = [&](Index c) -> bool {
    return sheet_component.size() > 0 && sheet_component[c];
  };

  // Mode-2 self-merge: open components emit `(2c, 2c+1)` so their
  // two sides collapse into one domain. Sheet components are exempt.
  auto open_mask = ag.open_component_mask();
  auto emit_open_merges = [&](tf::buffer<std::array<Index, 2>> &merges) {
    for (Index c = Index(0); c < n_components; ++c)
      if (open_mask[c] && !is_sheet_component(c))
        merges.push_back({Index(2 * c), Index(2 * c + 1)});
  };

  // ---- 1. NM edge fans from connectivity. ---------------------------
  out.fans = tf::cut::make_non_manifold_edge_fans(ag, fc);
  const auto n_nm_edges = out.fans.edges.size();

  if (n_nm_edges == 0) {
    // No non-manifold incidences → every closed component has its
    // two sides as distinct domains; open components self-merge via
    // boundary pairs. Every component is its own bundle.
    tf::buffer<std::array<Index, 2>> merges;
    emit_open_merges(merges);
    out.n_domains =
        tf::cut::build_domain_of_side(merges, n_components, out.domain_of_side);

    out.bundle_of_component.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_for_each(
        tf::enumerate(out.bundle_of_component),
        [](auto t) {
          auto &&[i, b] = t;
          b = Index(i);
        },
        tf::checked);
    out.n_bundles = n_components;
    std::tie(out.tag_of_component, out.bundle_to_tags) =
        tf::cut::compute_bundle_tag_index(ag, fc, out.bundle_of_component,
                                            out.n_bundles);
    return out;
  }

  // ---- 2. Path id + per-path edge alignment. ------------------------
  // Anchors per-path direction so canonical-permutation buckets do
  // not pool across disjoint polylines.
  tf::edge_path_connector<vertex_t, Index, arrangement_vertex_hash<Index>> epc;
  epc.build(tf::make_edges(out.fans.edges));

  tf::buffer<Index> edge_path_id;
  edge_path_id.allocate(static_cast<std::size_t>(n_nm_edges));
  tf::parallel_for_each(
      tf::enumerate(tf::zip(epc.paths(), epc.directions())),
      [&edge_path_id, &fans = out.fans](auto t) {
        auto &&[path_idx, pd] = t;
        auto &&[path, path_dirs] = pd;
        for (auto &&[e_id, fwd] : tf::zip(path, path_dirs)) {
          edge_path_id[e_id] = static_cast<Index>(path_idx);
          if (!fwd) {
            std::swap(fans.edges[e_id][0], fans.edges[e_id][1]);
            // Occurrence directions are relative to the edge as stored;
            // flipping the edge flips every occurrence.
            auto o0 = fans.faces.offsets_buffer()[e_id];
            auto o1 = fans.faces.offsets_buffer()[e_id + 1];
            for (auto k = o0; k < o1; ++k)
              fans.dirs[k] ^= char(1);
          }
        }
      },
      tf::checked);

  // ---- 3. Canonicalize fans (radial sort + set keys). ---------------
  tf::buffer<Index> id_sorted_labels;
  id_sorted_labels.allocate(out.fans.faces.data_buffer().size());

  tf::buffer<char> is_valid;
  is_valid.allocate(static_cast<std::size_t>(n_nm_edges));
  tf::parallel_fill(is_valid, char(0));

  auto id_sorted_view = tf::make_offset_block_range(
      out.fans.faces.offsets_buffer(), id_sorted_labels);
  auto labels_indirect = tf::make_indirect_range(
      out.fans.faces.data_buffer(), ag.loop_labels());
  auto labels_view = tf::make_offset_block_range(
      out.fans.faces.offsets_buffer(), labels_indirect);

  auto dirs_view = tf::make_offset_block_range(
      out.fans.faces.offsets_buffer(), tf::make_range(out.fans.dirs));
  tf::cut::canonicalize_nm_edges<Int>(ag, fc, out.fans.edges, out.fans.faces,
                                       dirs_view, id_sorted_view, labels_view,
                                       is_valid, get_point, apply_to_face);

  // ---- 4. One rep per (path_id, set-key) bucket. --------------------
  out.reps = tf::cut::compute_majority_rep(Index(n_nm_edges), is_valid,
                                            edge_path_id, id_sorted_view,
                                            labels_view);

  // ---- 5. Wedge-radial merges + open-component boundary self-merges. ---
  tf::buffer<std::array<Index, 2>> merges;
  tf::buffer<std::array<Index, 2>> bundle_merges;
  tf::cut::emit_domain_merges(ag, out.fans, out.reps, merges,
                               bundle_merges);
  emit_open_merges(merges);

  // ---- 6. Union-find: domain_of_side + bundle_of_component. ---------
  out.n_domains =
      tf::cut::build_domain_of_side(merges, n_components, out.domain_of_side);

  auto bundle_labels =
      tf::cut::make_bundle_labels(bundle_merges, n_components);
  out.bundle_of_component = std::move(bundle_labels.labels);
  out.n_bundles = bundle_labels.n_components;

  std::tie(out.tag_of_component, out.bundle_to_tags) =
      tf::cut::compute_bundle_tag_index(ag, fc, out.bundle_of_component,
                                          out.n_bundles);

  return out;
}

} // namespace tf::cut
