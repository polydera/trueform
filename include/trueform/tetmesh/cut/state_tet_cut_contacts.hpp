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
#include "../../core/buffer.hpp"
#include "../../intersect/records/tagged_intersection.hpp"
#include "../../topology/topo_type.hpp"
#include "./state_tet_chord_crossings.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::tetmesh {

/// One side of a record between two traced triangles: the triangle `face`
/// (its position in the cut), its partner, the point named, where the point
/// stands on the triangle and on its partner, and whether the two are
/// coplanar.
template <typename Index> struct tet_cut_contact {
  Index face;
  Index partner;
  Index id;
  tf::topo_id<Index> target;
  tf::topo_id<Index> target_other;
  bool coplanar;
};

/// The contacts of the records `records` between the traced triangles
/// `cut`, both sides of each, sorted by face, partner and point into
/// `contacts`, each face's run in `face_contacts`; a contact of two
/// triangles of one facet `facet_of` names that only shares a corner states
/// nothing, and a record is RESOLVED where the identity its point speaks
/// (`speaks`) belongs to the contact's feature on both — a vertex itself,
/// an edge its two endpoints, a face its three corners: the two already
/// meet at that vertex, which the grid made one, so the contact names the
/// vertex and splits nothing. Every point on a triangle's edge
/// (`edge_key(t, k)`, slot pairs) enters `edge_points` (sorted `{a, b,
/// point}`), and every contact of a face with one partner that is
/// not coplanar and names two points is a chord of `chords`, `edge` the edge of
/// the face both its ends stand on. O(records log records).
template <typename Index, typename Records, typename Cut, typename PointOf,
          typename FacetOf, typename EdgeKey, typename CornersOf,
          typename Speaks>
auto state_tet_cut_contacts(
    const Records &records, const Cut &cut, const PointOf &point_of,
    const FacetOf &facet_of, const EdgeKey &edge_key,
    const CornersOf &corners_of, const Speaks &speaks,
    tf::buffer<tet_cut_contact<Index>> &contacts,
    tf::buffer<std::array<std::size_t, 2>> &face_contacts,
    tf::buffer<std::array<Index, 3>> &edge_points,
    tf::buffer<tet_cut_chord<Index>> &chords, tf::buffer<Index> &ids) -> void {
  contacts.clear();
  for (const auto &record : records.flat_intersections()) {
    if (record.object == record.object_other)
      continue;
    const Index ta = cut[std::size_t(record.object)];
    const Index tb = cut[std::size_t(record.object_other)];
    if (facet_of(ta) == facet_of(tb) &&
        record.target.label == tf::topo_type::vertex &&
        record.target_other.label == tf::topo_type::vertex)
      continue;
    Index id = point_of[std::size_t(record.id)];
    const Index vertex = speaks(id);
    const auto in_feature = [&](Index t, const tf::topo_id<Index> &feature) {
      const auto corners = corners_of(t);
      const auto at = [&](std::size_t k) {
        return speaks(corners[k]) == vertex;
      };
      if (feature.label == tf::topo_type::vertex)
        return at(std::size_t(feature.id));
      if (feature.label == tf::topo_type::edge)
        return at(std::size_t(feature.id)) ||
               at((std::size_t(feature.id) + 1) % 3);
      return at(0) || at(1) || at(2);
    };
    if (in_feature(ta, record.target) && in_feature(tb, record.target_other))
      for (auto corner : corners_of(ta))
        if (speaks(corner) == vertex)
          id = corner;
    const bool coplanar = record.flags & tf::intersect::coplanar_pair_flag;
    contacts.push_back({record.object, record.object_other, id, record.target,
                        record.target_other, coplanar});
    contacts.push_back({record.object_other, record.object, id,
                        record.target_other, record.target, coplanar});
  }
  std::sort(
      contacts.begin(), contacts.end(),
      [](const tet_cut_contact<Index> &a, const tet_cut_contact<Index> &b) {
        return std::make_tuple(a.face, a.partner, a.id) <
               std::make_tuple(b.face, b.partner, b.id);
      });
  face_contacts.allocate(cut.size());
  for (std::size_t f = 0, at = 0; f < cut.size(); ++f) {
    while (at < contacts.size() && std::size_t(contacts[at].face) < f)
      ++at;
    std::size_t end = at;
    while (end < contacts.size() && std::size_t(contacts[end].face) == f)
      ++end;
    face_contacts[f] = {at, end};
  }
  edge_points.clear();
  for (const auto &c : contacts)
    if (c.target.label == tf::topo_type::edge) {
      const auto key =
          edge_key(cut[std::size_t(c.face)], std::size_t(c.target.id));
      edge_points.push_back({key[0], key[1], c.id});
    }
  std::sort(edge_points.begin(), edge_points.end());
  edge_points.erase_till_end(
      std::unique(edge_points.begin(), edge_points.end()));
  const auto edge_along = [&](Index t, Index a, Index b) {
    const auto corners = corners_of(t);
    const auto on = [&](Index x, std::size_t k) {
      const auto key = edge_key(t, k);
      return x == corners[k] || x == corners[(k + 1) % 3] ||
             std::binary_search(edge_points.begin(), edge_points.end(),
                                std::array<Index, 3>{key[0], key[1], x});
    };
    for (std::size_t k = 0; k < 3; ++k)
      if (on(a, k) && on(b, k))
        return Index(k);
    return Index(-1);
  };
  chords.clear();
  for (std::size_t f = 0; f < cut.size(); ++f) {
    const auto [begin, end] = face_contacts[f];
    for (std::size_t c = begin; c < end;) {
      std::size_t h = c;
      while (h < end && contacts[h].partner == contacts[c].partner)
        ++h;
      ids.clear();
      bool coplanar = false;
      for (std::size_t k = c; k < h; ++k) {
        coplanar = coplanar || contacts[k].coplanar;
        ids.push_back(contacts[k].id);
      }
      std::sort(ids.begin(), ids.end());
      ids.erase_till_end(std::unique(ids.begin(), ids.end()));
      if (!coplanar && ids.size() == 2)
        chords.push_back({Index(f), contacts[c].partner, ids[0], ids[1],
                          edge_along(cut[f], ids[0], ids[1])});
      c = h;
    }
  }
}

} // namespace tf::tetmesh
