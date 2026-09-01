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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/domains/build_domain_of_side.hpp"
#include "../../topology/domains/make_bundle_labels.hpp"
#include "./arrangement_descriptor.hpp"
#include "./compute_bundle_tag_index.hpp"
#include "./emit_domain_merges.hpp"
#include "./make_plane_radial_fans.hpp"
#include "./triangle_component_labels.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Build the @ref tf::csg::graph::arrangement_descriptor for an
///        implicit N-form arrangement.
///
/// The wedge-merging half of the domain pipeline. The arrangement's
/// radial fans hold one fan per fan piece, its carrier planes already in
/// exact radial order and each page holding the live occurrences that sit
/// on it with their traversal bits
/// (@ref tf::csg::graph::make_plane_radial_fans). A constraint bounds the
/// cells, the fence decides which of them carries a fan, and the fan is
/// where the wedges are paired: nothing is elected, because one fan
/// already is one 1-cell. A fan needs two pages to pair — a lone page has
/// no ring to walk — and every admitted one is walked cyclically into
/// component-side and component-only merge pairs
/// (@ref tf::csg::graph::emit_domain_merges) that a dense union-find turns
/// into `domain_of_side` and `bundle_of_component`.
///
/// Nesting (domain–domain containment of disjoint shells) and
/// per-form sidedness classification are downstream stages that build
/// on this descriptor; this function does not perform them.
///
/// @tparam Int      Exact integer type for the radial tier's predicate
///                  intermediates.
///
/// @param get_mesh_point Callable `(int tag, Index id) ->
///        tf::point<Int, 3>` over a form's own point ids.
/// @param is_sheet_tag Per-form sheet mask (may be empty: no sheets).
///        Open components of sheet forms are exempt from the Mode-2
///        self-merge: a sheet's fragments always separate their two
///        sides, realising the virtual extension of an open cutter.
template <typename Int, typename Index, typename Arrangement,
          typename GetMeshPoint, typename ApplyToFace>
auto make_arrangement_descriptor(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    GetMeshPoint get_mesh_point, ApplyToFace apply_to_face,
    const tf::buffer<char> &is_sheet_tag = {})
    -> arrangement_descriptor<Index> {
  using labels_t = tf::csg::graph::triangle_component_labels<Index>;

  arrangement_descriptor<Index> out;
  const Index n_components = labels.n_components();
  const Index n_tags = arrangement.n_tags();

  // Components of sheet forms never self-merge; mark them before
  // compute_bundle_tag_index runs. Only sheet geometry is touched.
  tf::buffer<char> sheet_component;
  if (is_sheet_tag.size() > 0) {
    sheet_component.allocate(static_cast<std::size_t>(n_components));
    tf::parallel_fill(sheet_component, char(0));
    auto triangle_labels = labels.triangle_labels();
    auto triangle_tags = arrangement.triangle_tags();
    tf::parallel_for_each(
        tf::make_sequence_range(static_cast<Index>(triangle_labels.size())),
        [&](Index e) {
          const Index t = triangle_tags[e];
          if (t < Index(0) || t >= Index(is_sheet_tag.size()) ||
              !is_sheet_tag[t])
            return;
          const Index c = triangle_labels[e];
          if (c != labels_t::none_label)
            sheet_component[c] = char(1);
        });
    for (Index t = Index(0); t < n_tags; ++t) {
      if (t >= Index(is_sheet_tag.size()) || !is_sheet_tag[t])
        continue;
      auto polygon_labels = labels.polygon_labels(t);
      tf::parallel_for_each(
          tf::make_sequence_range(static_cast<Index>(polygon_labels.size())),
          [&](Index f) {
            const Index c = polygon_labels[f];
            if (c != labels_t::none_label)
              sheet_component[c] = char(1);
          });
    }
  }
  auto is_sheet_component = [&](Index c) -> bool {
    return sheet_component.size() > 0 && sheet_component[c];
  };

  // Mode-2 self-merge: open components emit `(2c, 2c+1)` so their
  // two sides collapse into one domain. Sheet components are exempt.
  auto open_mask = labels.open_component_mask();
  auto emit_open_merges = [&](tf::buffer<std::array<Index, 2>> &merges) {
    for (Index c = Index(0); c < n_components; ++c)
      if (open_mask[c] && !is_sheet_component(c))
        merges.push_back({Index(2 * c), Index(2 * c + 1)});
  };

  out.fans = tf::csg::graph::make_plane_radial_fans<Index, Int>(
      arrangement.arrangement(), arrangement.world(),
      arrangement.piece_incidence(), arrangement.piece_fences(),
      get_mesh_point, apply_to_face);
  const auto n_fans = out.fans.pieces.size();

  if (n_fans == 0) {
    // No non-manifold incidences → every closed component has its
    // two sides as distinct domains; open components self-merge via
    // boundary pairs. Every component is its own bundle.
    tf::buffer<std::array<Index, 2>> merges;
    emit_open_merges(merges);
    out.n_domains = tf::topology::domains::build_domain_of_side(
        merges, n_components, out.domain_of_side);

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
        tf::csg::graph::compute_bundle_tag_index(
            labels, n_tags, labels.triangle_labels(),
            arrangement.triangle_tags(), out.bundle_of_component,
            out.n_bundles);
    return out;
  }

  // One fan already is one 1-cell, so every admitted fan's wedge
  // relations enter the partition and nothing is elected or discarded.
  out.valid.allocate(static_cast<std::size_t>(n_fans));
  out.n_invalid_fans = 0;
  for (std::size_t fan = 0; fan < n_fans; ++fan) {
    const auto stands = out.fans.page_offsets[fan + 1] -
                            out.fans.page_offsets[fan] >=
                        Index(2);
    out.valid[fan] = char(stands);
    out.n_invalid_fans += Index(!stands);
  }

  tf::buffer<std::array<Index, 2>> merges;
  tf::buffer<std::array<Index, 2>> bundle_merges;
  tf::csg::graph::emit_domain_merges(labels.triangle_labels(), out.fans,
                                     arrangement.exposed_of_row(), out.valid,
                                     merges, bundle_merges);
  emit_open_merges(merges);

  out.n_domains = tf::topology::domains::build_domain_of_side(
      merges, n_components, out.domain_of_side);

  auto bundle_labels =
      tf::topology::domains::make_bundle_labels(bundle_merges, n_components);
  out.bundle_of_component = std::move(bundle_labels.labels);
  out.n_bundles = bundle_labels.n_components;

  std::tie(out.tag_of_component, out.bundle_to_tags) =
      tf::csg::graph::compute_bundle_tag_index(
          labels, n_tags, labels.triangle_labels(), arrangement.triangle_tags(),
          out.bundle_of_component, out.n_bundles);

  return out;
}

} // namespace tf::csg::graph
