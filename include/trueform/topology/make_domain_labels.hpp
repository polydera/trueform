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
#include "../core/algorithm/make_equivalence_class_map.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/checked.hpp"
#include "../core/buffer.hpp"
#include "../core/complete.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../exact_coordinate_converter.hpp"
#include "./domain_config.hpp"
#include "./domains/build_domain_of_side.hpp"
#include "./domains/canonicalize_nm_edges.hpp"
#include "./domains/compute_domain_volumes.hpp"
#include "./domains/compute_majority_rep.hpp"
#include "./domains/compute_volume_contributions_per_fragment.hpp"
#include "./domains/emit_boundary_merges.hpp"
#include "./domains/emit_domain_merges.hpp"
#include "./domains/filter_face_blocks.hpp"
#include "./domains/lift_to_domain_labels.hpp"
#include "./domains/make_bundle_labels.hpp"
#include "./domains/make_nesting_merges.hpp"
#include "./edge_path_connector.hpp"
#include "./make_face_membership.hpp"
#include "./make_manifold_edge_connected_component_labels.hpp"
#include "../exact/resolve_int_type.hpp"
#include "./make_manifold_edge_link.hpp"
#include "./non_manifold_edges.hpp"
#include "./policy/face_membership.hpp"
#include "./policy/manifold_edge_link.hpp"
namespace tf {

/// @ingroup topology_components
/// @brief Per-face per-side domain labels of a non-manifold polygon mesh.
///
/// A non-manifold surface mesh bounds multiple 3D regions ("domains").
/// Each face has two sides; each side bounds one domain. Returns a
/// @ref tf::domain_labels with one 2-block per face: `labels[f][0]` is
/// the domain id on the face's stored-orientation side, `labels[f][1]`
/// is the domain on the reversed side; `n_domains` is the total number
/// of distinct domains. Under
/// @ref tf::domain_config::ignore_open_fragments, faces in MEL
/// components carrying boundary edges are excluded from the partition
/// and carry the sentinel label `n_domains`.
///
/// @pre Faces must be consistently oriented within each manifold-edge
///      connected component (e.g. arrangement output, or call
///      @ref tf::orient_faces_consistently first). The per-face "side
///      0 / side 1" mapping is derived from local edge direction;
///      consistent winding makes it coherent across a fragment without
///      an explicit orientation flood.
///
/// @tparam Int Exact integer type for the angular comparator. Default
///         `tf::none_t` → resolved to int32 for float coords, int64 for
///         double coords, identity for integer coords (via
///         `tf::exact::resolve_int_type`). Predicate intermediates use
///         `tf::exact::meta<ResolvedInt>`.
/// @tparam Policy The polygons policy.
/// @param polygons The polygons range.
/// @param config @ref tf::domain_config flag set selecting open-fragment
///        handling and other behaviour.
template <typename Int = tf::none_t, typename Policy>
auto make_domain_labels(const tf::polygons<Policy> &polygons,
                        tf::domain_config config = tf::domain_config::none) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using CoordType = tf::coordinate_type<Policy>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, CoordType>;

  if constexpr (!tf::has_face_membership_policy<Policy>) {
    auto fm = tf::make_face_membership(polygons);
    return make_domain_labels<Int>(polygons | tf::tag(fm), config);
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
    auto mel = tf::make_manifold_edge_link(polygons);
    return make_domain_labels<Int>(polygons | tf::tag(mel), config);
  } else {
    auto fragment_labels =
        tf::make_manifold_edge_connected_component_labels(polygons);
    auto [nm_edges, nm_edge_faces] =
        tf::make_non_manifold_edges(polygons, tf::complete);

    // Path id + per-path edge-axis alignment anchors sa within a chain,
    // so canonical-permutation buckets do not pool across polylines.
    tf::edge_path_connector<Index, Index> epc;
    epc.build(tf::make_edges(nm_edges));
    tf::buffer<Index> edge_path_id;
    edge_path_id.allocate(nm_edges.size());
    tf::parallel_for_each(
        tf::enumerate(tf::zip(epc.paths(), epc.directions())),
        [&edge_path_id, &nm_edges = nm_edges](auto t) {
          auto &&[path_idx, pd] = t;
          auto &&[path, dirs] = pd;
          for (std::size_t pos = 0; pos < path.size(); ++pos) {
            Index e_id = path[pos];
            edge_path_id[e_id] = static_cast<Index>(path_idx);
            if (!dirs[pos])
              std::swap(nm_edges[e_id][0], nm_edges[e_id][1]);
          }
        },
        tf::checked);

    tf::buffer<std::array<Index, 2>> merges;
    tf::topology::domains::emit_boundary_merges(polygons, fragment_labels,
                                                merges);

    Index anchor_side = Index(-1);

    // Mode 1: filter open-component faces out of nm_edge_faces and
    // route all open components into one garbage equivalence class via
    // synthetic (anchor_side, 2c) connectors.
    if ((config & tf::domain_config::ignore_open_fragments) &&
        merges.size() > 0) {
      tf::buffer<char> open_component_mask;
      open_component_mask.allocate(fragment_labels.n_components);
      tf::parallel_fill(open_component_mask, char(0));
      tf::parallel_for_each(merges, [&](const auto &pair) {
        open_component_mask[pair[0] / 2] = char(1);
      });
      auto face_exclude_mask = tf::make_indirect_range(
          fragment_labels.labels, open_component_mask);
      nm_edge_faces = tf::topology::domains::filter_face_blocks(
          nm_edge_faces, face_exclude_mask);

      anchor_side = merges[0][0];
      auto n_open = merges.size();
      for (decltype(n_open) i = 1; i < n_open; ++i)
        merges.push_back({anchor_side, merges[i][0]});
    }

    // Float coords go through make_pt_converter (AABB-scaled to ~99%
    // of int_max); integer coords are identity-mapped. Same pattern as
    // hole_patcher / planar_graph_regions.
    auto get_point = [&]() {
      using CoordT = tf::coordinate_type<Policy>;
      if constexpr (std::is_integral_v<CoordT>) {
        return [&polygons](Index v) -> tf::point<ResolvedInt, 3> {
          auto p = polygons.points()[v];
          return tf::point<ResolvedInt, 3>{ResolvedInt(p[0]),
                                           ResolvedInt(p[1]),
                                           ResolvedInt(p[2])};
        };
      } else {
        auto conv =
            tf::make_exact_coordinate_converter<ResolvedInt>(polygons);
        return [conv, &polygons](Index v) -> tf::point<ResolvedInt, 3> {
          return conv(polygons.points()[v]);
        };
      }
    }();

    tf::buffer<Index> id_sorted_labels;
    id_sorted_labels.allocate(nm_edge_faces.data_buffer().size());
    tf::buffer<char> is_valid;
    is_valid.allocate(nm_edges.size());
    tf::parallel_fill(is_valid, char(0));
    auto id_sorted_view = tf::make_offset_block_range(
        nm_edge_faces.offsets_buffer(), id_sorted_labels);
    auto labels_indirect = tf::make_indirect_range(nm_edge_faces.data_buffer(),
                                                   fragment_labels.labels);
    auto labels_view = tf::make_offset_block_range(
        nm_edge_faces.offsets_buffer(), labels_indirect);

    tf::topology::domains::canonicalize_nm_edges<ResolvedInt>(
        polygons, fragment_labels, nm_edges, nm_edge_faces, id_sorted_view,
        labels_view, is_valid, get_point);

    auto reps = tf::topology::domains::compute_majority_rep(
        Index(nm_edges.size()), is_valid, edge_path_id, id_sorted_view,
        labels_view);

    tf::buffer<std::array<Index, 2>> bundle_merges;
    tf::topology::domains::emit_domain_merges(polygons, fragment_labels,
                                              nm_edges, nm_edge_faces, reps,
                                              merges, bundle_merges);

    tf::buffer<Index> domain_of_side;
    auto n_domains = tf::topology::domains::build_domain_of_side(
        merges, fragment_labels.n_components, domain_of_side);

    auto bundle_labels = tf::topology::domains::make_bundle_labels(
        bundle_merges, fragment_labels.n_components);

    auto volume_contributions =
        tf::topology::domains::compute_volume_contributions_per_fragment(
            polygons, fragment_labels);

    auto domain_volumes = tf::topology::domains::compute_domain_volumes(
        volume_contributions, domain_of_side, n_domains);

    Index removed_domain =
        anchor_side >= Index(0) ? domain_of_side[anchor_side] : Index(-1);
    tf::buffer<std::array<Index, 2>> nesting_merges;
    Index root_anchor = tf::topology::domains::make_nesting_merges<ResolvedInt>(
        polygons, fragment_labels, bundle_labels, domain_of_side,
        domain_volumes, get_point, nesting_merges, removed_domain);

    // exclude_outer_shell: lump the universe into the sentinel
    // equivalence class. Adopt the universe as removed_domain if Mode 1
    // didn't already establish one.
    if ((config & tf::domain_config::exclude_outer_shell) &&
        root_anchor >= Index(0)) {
      if (removed_domain >= Index(0) && removed_domain != root_anchor) {
        nesting_merges.push_back(
            {std::min(root_anchor, removed_domain),
             std::max(root_anchor, removed_domain)});
      } else if (removed_domain < Index(0)) {
        removed_domain = root_anchor;
      }
    }

    if (nesting_merges.size() > 0) {
      tf::buffer<Index> domain_remap;
      domain_remap.allocate(n_domains);
      n_domains = Index(tf::make_dense_equivalence_class_map(nesting_merges,
                                                             domain_remap));
      tf::parallel_for_each(
          domain_of_side, [&](Index &d) { d = domain_remap[d]; }, tf::checked);
      if (removed_domain >= Index(0))
        removed_domain = domain_remap[removed_domain];
    }

    return tf::topology::domains::lift_to_domain_labels(
        domain_of_side, n_domains, fragment_labels, Index(polygons.size()),
        removed_domain);
  }
}

} // namespace tf
