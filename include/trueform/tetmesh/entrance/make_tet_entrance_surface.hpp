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
#include "../../core/aabb_from.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/points.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/input_lattice.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../topology/make_face_membership.hpp"
#include "../../topology/make_manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "./count_tet_merged_open_edges.hpp"
#include "./make_tet_entrance_converter.hpp"
#include "./merge_tet_facets.hpp"
#include "./state_tet_entrance_surface.hpp"
#include "./state_tet_reach_cells.hpp"
#include "./tet_entrance_budget.hpp"
#include "./tet_entrance_surface.hpp"
#include "./tet_entrance_workspace.hpp"
#include "./tet_facet_workspace.hpp"
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Bring `polygons` onto the `Int` lattice once: the points, placed by the
/// frame `polygons` carries, convert at the input's own precision, rounded
/// once, onto the lattice @ref tet_entrance_budget states for the `Int` lattice
/// stored in `Coord`, fitted (@ref make_tet_entrance_converter) to the placed
/// box of the points the faces name, and the surface is stated on them
/// (@ref state_tet_entrance_surface): the input's triangles as given, their
/// points numbered by first appearance through the faces, the points in one
/// cell of the reach grid one point (@ref state_tet_reach_cells), the
/// membership built for that numbering and a carried link, whose peers are
/// faces, read as it stands where no point joined another. Under a
/// `tolerance` in the input's coordinates the door's pooling places the
/// points (@ref tf::exact::input_lattice::build_pooled, on that converter,
/// over the form and its face membership, which is all the pooling reads):
/// near-coplanar faces pool onto one plane, and the band it ran at is the
/// one every drain states its contacts under; at zero nothing of the door
/// executes. Crossings are the recovery's to find and the drains' to resolve
/// (@ref cut_tet_pairs).
template <typename Index, typename Int, typename Coord, typename Policy>
auto make_tet_entrance_surface(const tf::polygons<Policy> &polygons,
                               double tolerance) {
  using RealT = tf::coordinate_type<Policy>;
  tet_entrance_surface<Index, RealT, Int> surface;
  tet_entrance_workspace<Index, Int> work;
  const auto state_surface = [&surface,
                              &work](const auto &rows, const auto &sources,
                                     const auto &membership, const auto &link) {
    const auto peers = [&link](std::size_t t) { return link[t]; };
    std::array<std::size_t, 3> thrown{};
    Index n_facets = 0;
    {
      // the merge's tables end here, before the surface's are built
      tet_facet_workspace<Index, Int> merge;
      n_facets = merge_tet_facets<Int, Index>(
          surface.points.points(), rows, membership, peers, merge,
          work.triangles, work.origin, work.facet_of, thrown);
      surface.open_edges = count_tet_merged_open_edges(
          rows, membership, peers, work.triangles, thrown, merge.steps);
    }
    state_tet_entrance_surface(surface, n_facets, thrown, sources, peers, work,
                               surface.link);
  };
  const auto points = polygons.points();
  const auto pose = tf::frame_of(polygons);
  surface.input_bits = tet_entrance_budget<Int, Coord>::lattice_bits;
  const auto faces = polygons.faces();
  // first appearance is a scan over the corners, the one serial pass of the
  // numbering: a parallel first corner per point costs an atomic minimum per
  // corner and a second pass
  tf::buffer<Index> point_of;
  point_of.allocate(points.size());
  tf::parallel_fill(point_of, Index(-1));
  tf::buffer<Index> ids;
  ids.reserve(points.size());
  tf::blocked_buffer<Index, 3> numbered;
  numbered.allocate(faces.size());
  for (std::size_t f = 0; f < faces.size(); ++f)
    for (std::size_t k = 0; k < 3; ++k) {
      auto &at = point_of[std::size_t(faces[f][k])];
      if (at < 0) {
        at = Index(ids.size());
        ids.push_back(Index(faces[f][k]));
      }
      numbered[f][k] = at;
    }
  surface.converter = make_tet_entrance_converter<Int>(
      tf::transformed(
          tf::aabb_from(tf::make_points(tf::make_indirect_range(ids, points))),
          pose),
      surface.input_bits);
  surface.points.allocate(ids.size());
  const auto place = [&surface, &ids](const auto &position) {
    tf::parallel_for_each(
        tf::make_sequence_range(ids.size()),
        [&surface, &ids, &position](std::size_t i) {
          surface.points[i] = position(ids[i]);
        },
        tf::checked);
  };
  if (tolerance > 0.0) {
    const auto pool = [&](const auto &form) {
      const auto apply_to_form = [&form](Index, const auto &f) { f(form); };
      tf::exact::input_lattice<Index, RealT, Int> door;
      door.build_pooled(
          tf::exact::vertex_converter<Int, RealT, 3>{surface.converter},
          apply_to_form, Index(1), tolerance);
      surface.band = door.tolerance_int();
      const auto read = door.reader(apply_to_form);
      place([&read](Index id) { return read(0, id); });
    };
    if constexpr (tf::has_face_membership_policy<Policy>)
      pool(polygons);
    else {
      const auto membership = tf::make_face_membership<Index>(polygons);
      pool(polygons | tf::tag(membership));
    }
  } else
    place([&surface, &points, &pose](Index id) {
      return surface.converter(tf::transformed(points[std::size_t(id)], pose));
    });
  const std::size_t joined = state_tet_reach_cells(
      surface.points, numbered, Int(surface.converter.quantum()), work);
  const auto renumbered = tf::make_polygons(numbered, surface.points.points());
  const auto membership = tf::make_face_membership<Index>(renumbered);
  const auto sources = tf::make_sequence_range(faces.size());
  const auto state = [&](const auto &link) {
    state_surface(numbered, sources, membership, link);
  };
  const auto made = [&] {
    state(tf::make_manifold_edge_link<Index>(renumbered | tf::tag(membership)));
  };
  if constexpr (tf::has_manifold_edge_link_policy<Policy>) {
    if (joined)
      made();
    else
      state(polygons.manifold_edge_link());
  } else
    made();
  return surface;
}

} // namespace tf::tetmesh
