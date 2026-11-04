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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/make_face_membership.hpp"
#include "../../topology/make_manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "./arrange_tet_entrance.hpp"
#include "./make_tet_entrance_converter.hpp"
#include "./tet_entrance_budget.hpp"
#include "./merge_tet_facets.hpp"
#include "./tet_entrance_surface.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <tuple>

namespace tf::tetmesh {

/// Bring `polygons` onto the `Int` lattice once. The points convert at the
/// input's own precision, rounded once; the
/// self-arrangement of the converted polygons names every crossing by its
/// canonical identity and cuts the faces there, rounding each created point
/// once onto the same lattice, and is arranged again until a pass rounds
/// nothing (@ref arrange_tet_entrance), so the triangles it states are
/// compatible constraints; they are merged into facets by
/// @ref merge_tet_facets through the membership and edge link the
/// arrangement was handed, which throws out the degenerate ones it meets,
/// and ordered facet by facet. A pass that leaves a carrier without its
/// triangles ends the surface there, stating the carriers and faces and no
/// points or triangles. The lattice is the one @ref tet_entrance_budget
/// states for the `Int` lattice stored in `Coord`.
template <typename Index, typename Int, typename Coord, typename Policy>
auto make_tet_entrance_surface(const tf::polygons<Policy> &polygons) {
  using RealT = tf::coordinate_type<Policy>;
  tet_entrance_surface<Index, RealT, Int> surface;
  const auto points = polygons.points();
  surface.input_bits = tet_entrance_budget<Int, Coord>::lattice_bits;
  surface.converter =
      make_tet_entrance_converter<Int>(points, surface.input_bits);
  tf::points_buffer<Int, 3> lattice;
  lattice.allocate(points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(points.size()),
      [&lattice, &points, &surface](std::size_t i) {
        lattice[i] = surface.converter(points[i]);
      },
      tf::checked);
  const auto arranged = arrange_tet_entrance(
      tf::make_polygons(polygons.faces(), lattice.points()));
  const auto &mesh = std::get<0>(arranged);
  const auto &faces = std::get<1>(arranged);
  surface.passes = std::get<2>(arranged);
  surface.failed_planes = std::get<3>(arranged)[0];
  surface.failed_faces = std::get<3>(arranged)[1];
  if (surface.failed_faces)
    return surface;
  const auto cut = mesh.points();
  surface.points.allocate(cut.size());
  tf::parallel_for_each(
      tf::make_sequence_range(cut.size()),
      [&surface, &cut](std::size_t i) {
        for (std::size_t k = 0; k < 3; ++k)
          surface.points[i][k] = cut[i][k];
      },
      tf::checked);
  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<Index> origin;
  tf::buffer<Index> facet_of;
  std::array<std::size_t, 3> thrown{};
  const Index n_facets = merge_tet_facets<Int, Index>(
      cut, mesh.faces(), std::get<4>(arranged), std::get<5>(arranged),
      triangles, origin, facet_of, thrown, surface.open_edges);
  surface.dropped_degenerate = thrown[0] + thrown[1];
  surface.stitched_triangles = thrown[2];
  auto &offsets = surface.facets;
  offsets.allocate(std::size_t(n_facets) + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (auto facet : facet_of)
    ++offsets[std::size_t(facet) + 1];
  for (std::size_t f = 0; f < std::size_t(n_facets); ++f)
    offsets[f + 1] += offsets[f];
  tf::buffer<Index> slot;
  slot.allocate(triangles.size());
  {
    tf::buffer<Index> next;
    next.allocate(std::size_t(n_facets));
    std::copy(offsets.begin(), offsets.end() - 1, next.begin());
    for (std::size_t t = 0; t < triangles.size(); ++t)
      slot[t] = next[std::size_t(facet_of[t])]++;
  }
  surface.triangles.allocate(triangles.size());
  surface.parents.allocate(triangles.size());
  tf::parallel_for_each(
      tf::make_sequence_range(triangles.size()),
      [&surface, &triangles, &origin, &faces, &slot](std::size_t t) {
        const auto at = std::size_t(slot[t]);
        for (std::size_t k = 0; k < 3; ++k)
          surface.triangles[at][k] = triangles[t][k];
        surface.parents[at] = Index(faces[std::size_t(origin[t])]);
      },
      tf::checked);
  if (surface.dropped_degenerate || surface.stitched_triangles) {
    const auto membership = tf::make_face_membership<Index>(
        tf::make_polygons(surface.triangles, surface.points.points()));
    surface.link = tf::make_manifold_edge_link<Index>(
        tf::make_polygons(surface.triangles, surface.points.points()) |
        tf::tag(membership));
  } else {
    const auto &arranged_link = std::get<5>(arranged);
    surface.link.data_buffer().allocate(triangles.size() * 3);
    tf::parallel_for_each(
        tf::make_sequence_range(triangles.size()),
        [&surface, &arranged_link, &slot](std::size_t t) {
          const auto peers = arranged_link[t];
          auto placed = surface.link[std::size_t(slot[t])];
          for (std::size_t k = 0; k < 3; ++k) {
            const Index p = peers[k].face_peer;
            placed[k] = {p >= 0 ? slot[std::size_t(p)] : p};
          }
        },
        tf::checked);
  }
  return surface;
}

} // namespace tf::tetmesh
