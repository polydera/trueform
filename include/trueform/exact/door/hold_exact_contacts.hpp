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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../vertex_converter.hpp"
#include "./commit_contact_planes.hpp"
#include "./gather_exact_contacts.hpp"
#include "./make_support_plane.hpp"
#include "./place_vertices.hpp"
#include "./placement_census.hpp"
#include "./placement_tables.hpp"
#include "./point_on_face.hpp"
#include "./pool/pool_input_planes.hpp"
#include "./segments_touch.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// The exact identifications every vertex keeps, as the tables' held CSR:
/// a face it stands on exactly, and the plane an exact edge crossing of its
/// spans — the two crossing edges are coplanar, so their four ends ride one
/// plane and the lines keep meeting.
template <typename Index, typename Int, typename RealType>
auto state_held_identifications(placement_tables<Index, Int, RealType> &tables,
                                const exact_contacts<Index> &contacts) -> void {
  const auto n = tables.points.size();
  tables.held_planes.clear();
  tf::buffer<std::array<Index, 2>> rows;
  for (const auto &c : contacts.on_face)
    rows.push_back({c[0], c[1]});
  for (const auto &e : contacts.edges) {
    const std::array<tf::point<Int, 3>, 4> ends{
        tables.points[std::size_t(e[0])], tables.points[std::size_t(e[1])],
        tables.points[std::size_t(e[2])], tables.points[std::size_t(e[3])]};
    tf::exact::canonical_plane<Int> plane{};
    make_support_plane<Int>(ends, plane);
    const auto row = Index(-1) - Index(tables.held_planes.size());
    tables.held_planes.push_back(plane);
    for (const auto v : e)
      rows.push_back({v, row});
  }
  tables.held_offsets.allocate(n + 1);
  std::fill(tables.held_offsets.begin(), tables.held_offsets.end(), Index(0));
  for (const auto &r : rows)
    ++tables.held_offsets[std::size_t(r[0]) + 1];
  for (std::size_t v = 1; v <= n; ++v)
    tables.held_offsets[v] += tables.held_offsets[v - 1];
  tables.held.allocate(rows.size());
  tf::buffer<Index> at;
  at.allocate(n);
  std::copy(tables.held_offsets.begin(), tables.held_offsets.end() - 1,
            at.begin());
  for (const auto &r : rows)
    tables.held[std::size_t(at[std::size_t(r[0])]++)] = r[1];
}

/// THE PRIORITY RULE: an existing identification outranks a new one. The
/// planes settle and the vertices ride them with every exact identification
/// the input states in their memberships; then every one of those is asked
/// again, exactly, of the moved mesh. A vertex whose identification the move
/// undid is pinned to its input point — the new pull it took is forgone, the
/// old contact stands — and the placement runs again. A pin only ever adds,
/// and an edge a newly moved vertex carries is asked once, so the rounds
/// end; the last one breaks nothing the input stated.
template <typename Index, typename Int, typename RealType, typename ApplyToForm>
auto hold_exact_contacts(
    placement_tables<Index, Int, RealType> &tables,
    const pool::input_pools<Int> &pools,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    const ApplyToForm &apply_to_form, Index n_tags,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<Index> &face_offsets, Int tolerance,
    tf::buffer<Index> &ticket, tf::buffer<tf::point<Int, 3>> &moved,
    placement_census &census) -> void {
  const auto n = tables.points.size();
  exact_contacts<Index> contacts;
  gather_face_contacts(tables, contacts);

  // the vertices a contact can move: the admitted and the corners of the
  // faces they were admitted against. A vertex anything else moves — a pool —
  // is asked after the placement that moved it.
  tf::buffer<char> scanned;
  scanned.allocate(n);
  tf::parallel_fill(scanned, char(0));
  for (std::size_t v = 0; v < n; ++v) {
    if (tables.contact_offsets[v] == tables.contact_offsets[v + 1])
      continue;
    scanned[v] = char(1);
    for (auto k = tables.contact_offsets[v]; k < tables.contact_offsets[v + 1];
         ++k)
      for (const auto c :
           tables.face_corners(std::size_t(tables.contacts[std::size_t(k)])))
        scanned[std::size_t(c)] = char(1);
  }
  gather_edge_contacts(tables, converter, apply_to_form, n_tags, vertex_offsets,
                       face_offsets, scanned, contacts);

  struct verify_t {
    tf::small_vector<tf::point<Int, 3>, 8> corners;
  };
  tf::buffer<char> pinned;
  tf::buffer<char> fresh;
  tf::buffer<Index> breaks;
  std::int64_t rounds = 0;
  const auto at = [&tables, &ticket, &moved](Index v) {
    if (ticket.size() == 0 || ticket[std::size_t(v)] == Index(-1))
      return tables.points[std::size_t(v)];
    return moved[std::size_t(ticket[std::size_t(v)])];
  };
  for (;;) {
    ++rounds;
    state_held_identifications(tables, contacts);
    placement_census round = census;
    committed_planes<Index, Int> committed;
    commit_contact_planes(tables, pools, tolerance, pinned, committed, round);
    place_vertices(tables, committed, tolerance, pinned, ticket, moved, round);
    census = round;

    // an edge a newly moved vertex carries is asked before anything is
    // judged, so the kept set is whole for this placement
    fresh.allocate(n);
    bool unscanned = false;
    for (std::size_t v = 0; v < n; ++v) {
      fresh[v] =
          char(ticket.size() != 0 && ticket[v] != Index(-1) && !scanned[v]);
      unscanned = unscanned || fresh[v];
    }
    if (unscanned) {
      const auto before = contacts.edges.size();
      gather_edge_contacts(tables, converter, apply_to_form, n_tags,
                           vertex_offsets, face_offsets, fresh, contacts);
      for (std::size_t v = 0; v < n; ++v)
        scanned[v] = char(scanned[v] || fresh[v]);
      if (contacts.edges.size() != before)
        continue;
    }

    // every contact the placement undid names the vertices it holds, and
    // the moved ones among them are pinned after the barrier
    breaks.clear();
    tf::generic_generate(
        contacts.on_face, breaks,
        [&tables, &at](const std::array<Index, 2> &c, tf::buffer<Index> &into,
                       verify_t &local) {
          local.corners.clear();
          for (const auto k : tables.face_corners(std::size_t(c[1])))
            local.corners.push_back(at(k));
          if (point_on_face(at(c[0]), local.corners))
            return;
          into.push_back(c[0]);
          for (const auto k : tables.face_corners(std::size_t(c[1])))
            into.push_back(k);
        },
        verify_t{}, tf::checked);
    tf::generic_generate(
        contacts.edges, breaks,
        [&at](const std::array<Index, 4> &e, tf::buffer<Index> &into) {
          if (segments_touch<Int>(at(e[0]), at(e[1]), at(e[2]), at(e[3])))
            return;
          for (const auto v : e)
            into.push_back(v);
        },
        tf::checked);
    bool pin = false;
    for (const auto v : breaks) {
      if (ticket.size() == 0 || ticket[std::size_t(v)] == Index(-1))
        continue;
      if (pinned.size() == 0) {
        pinned.allocate(n);
        tf::parallel_fill(pinned, char(0));
      }
      if (!pinned[std::size_t(v)]) {
        pinned[std::size_t(v)] = char(1);
        pin = true;
      }
    }
    if (!pin)
      break;
  }
  census.kept_on_face = std::int64_t(contacts.on_face.size());
  census.kept_edges = std::int64_t(contacts.edges.size());
  census.rounds = rounds;
  std::int64_t count = 0;
  for (const auto p : pinned)
    count += p ? 1 : 0;
  census.pinned = count;
}

} // namespace tf::exact::door
