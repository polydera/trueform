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
#include "../../arrangement/identity/identity_substrate.hpp"
#include "../../arrangement/planes/plane_arrangement.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/topo_type.hpp"
#include "./plane_drain_world.hpp"
#include "./state_tet_chord_crossings.hpp"
#include "./state_tet_cut_contacts.hpp"
#include "./state_tet_lane_occurrences.hpp"
#include "./tet_changed_triangle.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::tetmesh {

/// The scratch @ref wave_tet_lanes walks on: the drain's world, its
/// definition tables, the plane arrangement whose wave splits it and the
/// identities the wave creates, named back into the drain.
template <typename Index, typename Int> struct tet_wave_workspace {
  tf::buffer<Index> member_offsets;
  tf::buffer<std::array<Index, 3>> members;
  tf::buffer<Index> member_triangle;
  tf::buffer<Index> member_cut;
  tf::buffer<Index> ring;
  tf::buffer<char> carrier_failed;
  tf::buffer<Index> face_def_offsets;
  tf::buffer<Index> emission_to_canon;
  tf::buffer<Index> run;
  tf::buffer<Index> side;
  tf::buffer<std::array<Index, 2>> chord_pieces;
  tf::intersect::graph::plane_tables<Index, Int> tables;
  tf::buffer<Index> vertex_offsets;
  tet_lane_occurrences<Index, Int> lane_points;
  tf::arrangement::identity_substrate<Index, Int> identities;
  tf::arrangement::plane_arrangement<Index, Int> arrangement;
  tf::buffer<tf::exact::pt3<Int>> created_point;
  tf::buffer<std::array<Index, 2>> no_pairs;
  tf::buffer<Index> created_name;
};

/// Split the lanes `due` of a drain on the plane arrangement's wave over
/// the drain's world (@ref plane_drain_world): every due lane one carrier —
/// lanes below `n_groups` the traced groups (positions in `cut`), the rest
/// one triangle each — and every live triangle beyond them holding an edge
/// of theirs (`holders`, `live`, `holds`) one carrier more, so a split the
/// wave lands on a lane's edge lands on a carrier the world holds. A
/// carrier states its members' sides split at the points on them
/// (`points_on_edge`) and their chords' pieces (`chord_points`); a carrier
/// whose triangulation preserves it ends its work, one that refuses is
/// resolved, its crossings and landings closed into identities and splits
/// carried to every carrier holding the edge, until nothing new is stated.
/// The identities the wave mints in one box of side `reach`, the input
/// quantum, are one (@ref
/// tf::arrangement::plane_arrangement::weld_near_identities); the wave's
/// identities are named by the drain's `authority` at their positions, a new
/// one appended to `points` (@ref tet_identity_authority), and every weld
/// the wave states is welded there. Every point is one flat identity, so a
/// side two carriers share is one set of definition rows and a split carries
/// across them; only a carrier's chart reads a point at its lane's occurrence
/// (@ref state_tet_lane_occurrences), every other tier at the position it was
/// stated at, and the pieces speak each
/// point's survivor, a piece two of whose corners became one dropped: identity
/// after geometry. Every triangle a
/// carrier gives pieces other than itself enters `changed`, ascending, its
/// pieces in `pieces`, none where every piece of it dropped, and every point
/// the wave states on a member's side enters `side_points_stated` as
/// `{a, b, point}`, so every reader of the drain's side points reads it; a
/// carrier still
/// refusing at the wave's fixed point, or one with a point inside a member that
/// no row of it ends at, is failed. Returns the carriers failed. O(world rows
/// log world rows + the wave's own cost: one preserve per carrier, a resolve
/// per refusal, each round over the carriers its statements reached).
template <typename Index, typename Int, typename Cut, typename CornersOf,
          typename EdgeKey, typename RowsOnEdge, typename SlotPoint,
          typename Point, typename Live, typename Holds, typename Authority>
auto wave_tet_lanes(
    tf::points_buffer<Int, 3> &points, Int reach, const tf::buffer<Index> &due,
    std::size_t n_groups, const tf::buffer<Index> &unit_runs,
    const tf::buffer<Index> &unit_members,
    const tf::buffer<std::array<Index, 2>> &neighbour_units, const Cut &cut,
    const CornersOf &corners_of, const EdgeKey &edge_key,
    const RowsOnEdge &rows_on_edge,
    const tf::buffer<std::array<Index, 3>> &side_rows,
    const tf::buffer<typename tf::exact::meta<Int>::param_type> &side_t,
    const SlotPoint &slot_point,
    const tf::buffer<tet_cut_contact<Index>> &contacts,
    const tf::buffer<std::array<std::size_t, 2>> &face_contacts,
    const tf::buffer<tet_cut_chord<Index>> &chords,
    const tf::buffer<std::array<Index, 3>> &chord_points, const Point &point,
    const tet_slot_holders<Index> &holders, const Live &live,
    const Holds &holds, tf::buffer<tet_changed_triangle<Index>> &changed,
    tf::buffer<std::array<Index, 3>> &pieces,
    tf::buffer<std::array<Index, 3>> &side_points_stated, Authority &authority,
    tet_wave_workspace<Index, Int> &work) -> std::size_t {
  auto &member_offsets = work.member_offsets;
  auto &members = work.members;
  auto &member_triangle = work.member_triangle;
  auto &member_cut = work.member_cut;
  member_offsets.clear();
  members.clear();
  member_triangle.clear();
  member_cut.clear();
  member_offsets.push_back(Index(0));
  for (auto unit : due) {
    const bool traced = std::size_t(unit) < n_groups;
    for (Index i = unit_runs[std::size_t(unit)];
         i < unit_runs[std::size_t(unit) + 1]; ++i) {
      const Index member = unit_members[std::size_t(i)];
      const Index t = traced ? cut[std::size_t(member)] : member;
      members.push_back(corners_of(t));
      member_triangle.push_back(t);
      member_cut.push_back(traced ? member : Index(-1));
    }
    member_offsets.push_back(Index(members.size()));
  }
  const std::size_t n_lanes = due.size();
  const std::size_t n_lane_members = members.size();
  // the ring: every live triangle holding an edge of a lane's triangle that
  // no lane holds
  auto &ring = work.ring;
  ring.clear();
  const auto in_lane = [&](Index u) {
    return std::binary_search(cut.begin(), cut.end(), u) ||
           std::binary_search(
               neighbour_units.begin(), neighbour_units.end(),
               std::array<Index, 2>{u, Index(-1)},
               [](const std::array<Index, 2> &x,
                  const std::array<Index, 2> &y) { return x[0] < y[0]; });
  };
  for (std::size_t f = 0; f < n_lane_members; ++f)
    for (std::size_t k = 0; k < 3; ++k) {
      const auto key = edge_key(member_triangle[f], k);
      for_each_tet_slot_holder(holders, key[0], [&](Index u) {
        if (live(std::size_t(u)) && holds(u, key[1]) && !in_lane(u))
          ring.push_back(u);
      });
    }
  std::sort(ring.begin(), ring.end());
  ring.erase_till_end(std::unique(ring.begin(), ring.end()));
  for (auto u : ring) {
    members.push_back(corners_of(u));
    member_triangle.push_back(u);
    member_cut.push_back(Index(-1));
    member_offsets.push_back(Index(members.size()));
  }
  const std::size_t n_carriers = member_offsets.size() - 1;

  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto distance2 = [](const auto &a, const auto &b) {
    T2 sum(0);
    for (std::size_t d = 0; d < 3; ++d) {
      const T2 delta = T2(T1(b[d]) - T1(a[d]));
      sum = sum + delta * delta;
    }
    return sum;
  };
  const auto speaks = [&authority](Index id) { return authority.standing(id); };
  // the lanes' charts: per lane one position of each identity its sides
  // state (@ref state_tet_lane_occurrences); the identity is one flat point
  auto &lane_points = work.lane_points;
  state_tet_lane_occurrences<Int>(member_offsets, members, member_triangle,
                                  rows_on_edge, side_rows, side_t, slot_point,
                                  lane_points);
  const auto world_plane_of = [&member_offsets](std::size_t f) {
    return std::size_t(std::upper_bound(member_offsets.begin() + 1,
                                        member_offsets.end(), Index(f)) -
                       (member_offsets.begin() + 1));
  };
  const auto side_points = [&](std::size_t f, std::size_t k,
                               tf::buffer<Index> &out) {
    out.clear();
    const auto &corners = members[f];
    rows_on_edge(member_triangle[f], k, [&](std::size_t r) {
      const Index x = side_rows[r][2];
      if (x != corners[0] && x != corners[1] && x != corners[2])
        out.push_back(x);
    });
    std::sort(out.begin(), out.end());
    out.erase_till_end(std::unique(out.begin(), out.end()));
    const auto lane = Index(world_plane_of(f));
    const auto at = [&](Index x) {
      return lane_points.chart(lane, x, tf::exact::pt3<Int>(point(x)));
    };
    const auto origin = at(corners[k]);
    std::sort(out.begin(), out.end(), [&](Index x, Index y) {
      const auto dx = distance2(origin, at(x));
      const auto dy = distance2(origin, at(y));
      return dx < dy || (!(dy < dx) && x < y);
    });
  };
  // a chord's end that one of its member's sides states is that side's
  // point
  const auto chord_end = [&](std::size_t f, Index id) {
    const Index survivor = speaks(id);
    Index found = Index(-1);
    for (std::size_t k = 0; k < 3 && found < 0; ++k)
      rows_on_edge(member_triangle[f], k, [&](std::size_t r) {
        if (found < 0 && speaks(side_rows[r][2]) == survivor)
          found = side_rows[r][2];
      });
    return found >= 0 ? found : id;
  };
  auto &run = work.run;
  const auto chord_pieces = [&](std::size_t f,
                                tf::buffer<std::array<Index, 2>> &out) {
    out.clear();
    const Index member = member_cut[f];
    if (member < 0)
      return;
    for (auto at = std::lower_bound(chords.begin(), chords.end(), member,
                                    [](const tet_cut_chord<Index> &c,
                                       Index face) { return c.face < face; });
         at != chords.end() && at->face == member; ++at) {
      if (at->edge >= 0)
        continue;
      state_tet_chord_run<Int>(at->a, at->b, chord_points, point, run);
      for (std::size_t k = 0; k + 1 < run.size(); ++k) {
        const Index u = chord_end(f, run[k]), v = chord_end(f, run[k + 1]);
        if (u != v)
          out.push_back({u, v});
      }
    }
  };
  const tet_drain_names<Index> names{
      authority.n_originals(),
      Index(Index(points.size()) - authority.n_originals())};
  auto &created_name = work.created_name;
  created_name.clear();
  state_tet_drain_tables<Index, Int>(
      names, member_offsets, members, side_points, chord_pieces,
      work.face_def_offsets, work.emission_to_canon, work.side,
      work.chord_pieces, work.tables);

  auto &failed = work.carrier_failed;
  failed.allocate(n_carriers);
  std::fill(failed.begin(), failed.end(), char(0));
  // a point a member touches inside that no row of its carrier ends at is
  // one the carrier cannot state
  auto &ends = work.side;
  for (std::size_t c = 0; c < n_lanes; ++c) {
    ends.clear();
    for (auto e : work.tables.plane_edges(Index(c))) {
      const auto &def = work.tables.defs()[std::size_t(e)];
      ends.push_back(names.point(def.point_tag_0, def.point_0, created_name));
      ends.push_back(names.point(def.point_tag_1, def.point_1, created_name));
    }
    std::sort(ends.begin(), ends.end());
    for (Index f = member_offsets[c]; f < member_offsets[c + 1]; ++f) {
      const Index member = member_cut[std::size_t(f)];
      if (member < 0)
        continue;
      const auto [begin, end] = face_contacts[std::size_t(member)];
      for (std::size_t k = begin; k < end; ++k)
        if (contacts[k].target.label == tf::topo_type::face &&
            !std::binary_search(ends.begin(), ends.end(),
                                chord_end(std::size_t(f), contacts[k].id)))
          failed[c] = 1;
    }
  }

  plane_drain_world<Index, Int, tf::points_buffer<Int, 3>> world(
      points, member_offsets, members, work.tables, lane_points);
  const auto read = [&world, &names, &created_name](std::int16_t tag,
                                                    Index id) {
    return world.point(names.point(tag, id, created_name));
  };
  auto &arrangement = work.arrangement;
  arrange_tet_drain(world, names, reach, read, work.vertex_offsets,
                    work.identities, arrangement);
  for (auto plane : arrangement.failed())
    failed[std::size_t(plane)] = 1;

  // the wave's identities named by the drain's authority, at their
  // positions, for no pair: one standing there speaks a created identity,
  // and every other one mints, appended to `points`
  auto &created_point = work.created_point;
  auto &no_pairs = work.no_pairs;
  const std::size_t n_created = std::size_t(work.identities.n_plane_created());
  created_point.allocate(n_created);
  for (std::size_t id = 0; id < n_created; ++id)
    created_point[id] = arrangement.resolve_created_point(
        work.identities, Index(names.n_created + Index(id)), read, read);
  no_pairs.allocate(n_created);
  std::fill(no_pairs.begin(), no_pairs.end(),
            std::array<Index, 2>{Index(-1), Index(-1)});
  const Index first =
      authority.name_positions(created_point, no_pairs, created_name);
  for (Index id = first; id < authority.n_identities(); ++id)
    points.push_back(authority.position_of(id));
  // what the wave welds the authority states: a drain point or a created
  // identity the wave retired speaks the identity it elected
  for (const auto &row : work.identities.plane_merges())
    authority.weld(row[0] == Index(0)
                       ? row[1]
                       : names.point_of_created(row[1], created_name),
                   names.point_of_created(row[2], created_name));
  // the chart ran on every point at the position it was stated at; its
  // pieces speak each point's survivor, identity after geometry
  const auto name = [&](Index corner) {
    return authority.standing(names.point_of_flat(corner, created_name));
  };

  // every point the wave states on a member's side joins the drain's side
  // points, `{a, b, point}` in slots, read off the rows each plane it
  // changed resolves through
  side_points_stated.clear();
  const auto blocks = arrangement.plane_tickets();
  for (std::size_t plane = 0; plane < blocks.size(); ++plane) {
    const Index block = blocks[plane];
    if (block == Index(-1))
      continue;
    for (const auto &def : arrangement.current_plane_defs(world, block)) {
      if (def.side < 0)
        continue;
      const auto t = member_triangle[std::size_t(def.face)];
      const auto &own = members[std::size_t(def.face)];
      const auto key = edge_key(t, std::size_t(def.side));
      for (const auto &[tag, id] :
           {std::pair<std::int16_t, Index>{def.point_tag_0, def.point_0},
            std::pair<std::int16_t, Index>{def.point_tag_1, def.point_1}}) {
        const Index point = names.point(tag, Index(id), created_name);
        if (point != own[0] && point != own[1] && point != own[2])
          side_points_stated.push_back({key[0], key[1], point});
      }
    }
  }
  changed.clear();
  pieces.clear();
  std::size_t n_failed = 0;
  for (std::size_t c = 0; c < n_carriers; ++c)
    n_failed += failed[c] != 0;
  const auto triangles = arrangement.triangles();
  for (std::size_t f = 0; f < members.size(); ++f) {
    const Index begin = Index(pieces.size());
    const auto span = arrangement.face_range(Index(f));
    for (Index i = span[0]; i < span[1]; ++i) {
      const auto tri = triangles[std::size_t(i)];
      const std::array<Index, 3> piece{name(Index(tri[0])), name(Index(tri[1])),
                                       name(Index(tri[2]))};
      if (piece[0] != piece[1] && piece[1] != piece[2] && piece[0] != piece[2])
        pieces.push_back(piece);
    }
    const auto &own = members[f];
    bool itself = Index(pieces.size()) - begin == Index(1);
    if (itself) {
      const auto &piece = pieces.back();
      itself = false;
      for (std::size_t r = 0; r < 3; ++r)
        itself =
            itself || (piece[0] == own[r] && piece[1] == own[(r + 1) % 3] &&
                       piece[2] == own[(r + 2) % 3]);
    }
    if (itself) {
      pieces.erase_till_end(pieces.begin() + std::ptrdiff_t(begin));
      continue;
    }
    // a member every piece of which dropped states nothing: it is a line
    changed.push_back({member_triangle[f], begin, Index(pieces.size())});
  }
  std::sort(changed.begin(), changed.end(),
            [](const tet_changed_triangle<Index> &a,
               const tet_changed_triangle<Index> &b) {
              return a.triangle < b.triangle;
            });
  return n_failed;
}

} // namespace tf::tetmesh
