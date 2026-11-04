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
#include "../../arrangement/make_arrangement_graph.hpp"
#include "../../arrangement/make_arrangement_mesh.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../reindex/return_source_ids.hpp"
#include "../../topology/make_face_membership.hpp"
#include "../../topology/make_manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "./measure_tet_entrance_holes.hpp"
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf::tetmesh {

/// The self-arrangement of the lattice `polygons`, arranged again on its own
/// output until a pass creates no point or mints no new position: `(mesh,
/// sources, passes)`, each triangle's source the input face it cuts. A pass
/// rounds every point it creates onto the lattice once, and a rounded point
/// can carry one of its cut triangles across a neighbouring constraint; the
/// next pass names that crossing and its gate welds what coincides, so the
/// output holds no crossing once a pass rounds nothing. A crossing whose
/// rounding stands on a vertex already there is welded onto it and named
/// again by every pass after, the geometry unchanged, so a pass that mints
/// no new position is the fixpoint the lattice admits. A pass that creates nothing leaves
/// the output it read as it stands, so an input whose first pass rounded no
/// crossing into being keeps that pass's output verbatim, and a clean input
/// is arranged once. The passes are bounded by
/// `max_passes`; measured inputs settle within five. A pass whose product
/// has a hole (@ref measure_tet_entrance_holes) ends the passes, and its
/// `{planes, faces}` come back as `holes`, `{0, 0}` when every carrier holds
/// its product. Every pass arranges a surface tagged with its face
/// membership and manifold edge link, built here once and handed on: the
/// pass builds neither again, and the output's own `membership` and `link`
/// come back beside it. A pass that creates nothing emits its input's faces
/// in their order over its own numbering of the points, so its link, whose
/// peers are faces, comes back as it is and its membership is built for the
/// output; both are built for the output otherwise.
template <typename Policy>
auto arrange_tet_entrance(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  constexpr std::size_t max_passes = 8;
  auto membership = tf::make_face_membership(polygons);
  auto link = tf::make_manifold_edge_link(polygons | tf::tag(membership));
  auto graph = tf::make_arrangement_graph(polygons | tf::tag(membership) |
                                          tf::tag(link));
  auto holes = measure_tet_entrance_holes(graph, polygons);
  auto arranged = tf::make_arrangement_mesh(graph, tf::return_source_ids);
  std::size_t created = graph.created_points().size();
  std::size_t passes = 1;
  bool fresh = created == 0;
  while (created && !holes[1] && passes < max_passes) {
    const auto &mesh = std::get<0>(arranged);
    const auto &sources = std::get<1>(arranged);
    membership = tf::make_face_membership<Index>(mesh.polygons());
    link = tf::make_manifold_edge_link<Index>(mesh.polygons() |
                                              tf::tag(membership));
    auto again = tf::make_arrangement_graph(
        mesh.polygons() | tf::tag(membership) | tf::tag(link));
    ++passes;
    created = again.created_points().size();
    if (!created) {
      fresh = true;
      break;
    }
    fresh = false;
    holes = measure_tet_entrance_holes(again, mesh.polygons());
    auto next = tf::make_arrangement_mesh(again, tf::return_source_ids);
    const bool stationary =
        std::get<0>(next).points().size() == mesh.points().size();
    auto &next_sources = std::get<1>(next);
    tf::parallel_for_each(
        tf::make_sequence_range(next_sources.size()),
        [&next_sources, &sources](std::size_t t) {
          next_sources[t] = sources[std::size_t(next_sources[t])];
        },
        tf::checked);
    arranged = std::move(next);
    if (stationary)
      break;
  }
  {
    const auto &mesh = std::get<0>(arranged);
    membership = tf::make_face_membership<Index>(mesh.polygons());
    if (!fresh)
      link = tf::make_manifold_edge_link<Index>(mesh.polygons() |
                                                tf::tag(membership));
  }
  return std::make_tuple(std::move(std::get<0>(arranged)),
                         std::move(std::get<1>(arranged)), passes, holes,
                         std::move(membership), std::move(link));
}

} // namespace tf::tetmesh
