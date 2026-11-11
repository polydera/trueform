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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../exact/vertex.hpp"
#include "./count_tet_drain_member_openness.hpp"
#include "./count_tet_drain_t_junctions.hpp"
#include "./enter_tet_retired_holders.hpp"
#include "./land_tet_edge_chords.hpp"
#include "./land_tet_reach_points.hpp"
#include "./lift_tet_crossings.hpp"
#include "./name_tet_proposals.hpp"
#include "./name_tet_record_points.hpp"
#include "./restate_tet_cut_facets.hpp"
#include "./state_tet_chord_crossings.hpp"
#include "./state_tet_cut_contacts.hpp"
#include "./state_tet_lanes.hpp"
#include "./state_tet_pair_records.hpp"
#include "./state_tet_point_keys.hpp"
#include "./state_tet_side_parameters.hpp"
#include "./tet_drain_cut.hpp"
#include "./tet_pair_workspace.hpp"
#include "./tet_slot_holders.hpp"
#include "./trace_tet_crossings.hpp"
#include "./wave_tet_lanes.hpp"
#include "./weld_tet_side_points.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Cut the crossings `named`, as the recovery's walk named them, on the
/// surface of lattice `points` and `triangles` — `faces` the same triangles
/// in site slots, blocked per facet at `facet_offsets`, each triangle's
/// input face in `parents`, `live(t)` the surface's
/// triangles, `segments` the recovery's segments in slots. The drain is its
/// steps, in order: the crossings lifted to triangle pairs
/// (@ref lift_tet_crossings), the curves through them traced
/// (@ref trace_tet_crossings), their records stated
/// (@ref state_tet_pair_records, at the band `band`) and named
/// (@ref name_tet_record_points),
/// the contacts, edge points and chords stated
/// (@ref state_tet_cut_contacts), the chord crossings and landings
/// (@ref state_tet_chord_crossings) named (@ref name_tet_proposals), the
/// chords along an edge landed on it (@ref land_tet_edge_chords), the lanes
/// stated (@ref state_tet_lanes) and split on the plane arrangement's wave
/// (@ref wave_tet_lanes), and the changed facets restated
/// (@ref restate_tet_cut_facets), every point the drain creates named by its
/// identity `authority` (@ref tet_identity_authority), a record's for the
/// pair of input faces it was found for (@ref state_tet_point_keys). A
/// drain that would leave a T-junction (@ref count_tet_drain_t_junctions),
/// or a member its pieces do not run once
/// (@ref count_tet_drain_member_openness), is failed.
/// `work` is the caller's scratch.
template <typename Index, typename Int, typename Triangles, typename Faces,
          typename FacetOffsets, typename Parents, typename Live,
          typename Segments, typename Named, typename Authority>
auto cut_tet_pairs(
    tf::points_buffer<Int, 3> &points, const Triangles &triangles,
    const Faces &faces, std::size_t n_slots, const FacetOffsets &facet_offsets,
    const Parents &parents, const Live &live, const Segments &segments,
    const Named &named, const tf::exact::pt_converter<Int, Int, 3> &identity,
    double band, tf::buffer<Index> &retired, tf::blocked_buffer<Index, 3> &rows,
    tf::buffer<Index> &sources, tf::buffer<std::array<Index, 2>> &beside,
    Authority &authority, tet_pair_workspace<Index, Int> &work)
    -> tet_drain_cut {
  using pt_t = tf::exact::pt3<Int>;
  tet_drain_cut drain;
  retired.clear();
  rows.clear();
  sources.clear();
  beside.clear();
  state_tet_slot_holders<Index>(faces, n_slots, work.holders);
  const std::size_t n_blocks = facet_offsets.size() - 1;
  const auto facet_of = [&](Index t) {
    return Index(std::upper_bound(
                     facet_offsets.begin(),
                     facet_offsets.begin() + std::ptrdiff_t(n_blocks + 1), t) -
                 facet_offsets.begin() - 1);
  };
  const auto holds = [&faces](Index t, Index slot) {
    const auto &f = faces[std::size_t(t)];
    return f[0] == slot || f[1] == slot || f[2] == slot;
  };
  const auto corners_of = [&](Index t) {
    return std::array<Index, 3>{Index(triangles[std::size_t(t)][0]),
                                Index(triangles[std::size_t(t)][1]),
                                Index(triangles[std::size_t(t)][2])};
  };
  const auto edge_key = [&faces](Index t, std::size_t k) {
    const Index a = faces[std::size_t(t)][k];
    const Index b = faces[std::size_t(t)][(k + 1) % 3];
    return std::array<Index, 2>{std::min(a, b), std::max(a, b)};
  };

  const auto holders_of = [&](Index s, tf::buffer<Index> &out) {
    out.clear();
    const auto &ends = segments[std::size_t(s)];
    for_each_tet_slot_holder(work.holders, Index(ends[0]), [&](Index u) {
      if (live(std::size_t(u)) && holds(u, ends[1]))
        out.push_back(u);
    });
  };
  const auto facet_triangles = [&](Index f, tf::buffer<Index> &out) {
    out.clear();
    for (Index t = facet_offsets[std::size_t(f)];
         t < facet_offsets[std::size_t(f) + 1]; ++t)
      if (live(std::size_t(t)))
        out.push_back(t);
  };
  const auto box_of = [&](Index t) {
    std::array<std::array<Int, 3>, 2> box{};
    for (std::size_t k = 0; k < 3; ++k) {
      const auto &p = points[std::size_t(triangles[std::size_t(t)][k])];
      for (std::size_t d = 0; d < 3; ++d) {
        box[0][d] = k ? std::min(box[0][d], Int(p[d])) : Int(p[d]);
        box[1][d] = k ? std::max(box[1][d], Int(p[d])) : Int(p[d]);
      }
    }
    return box;
  };
  lift_tet_crossings<Index, Int>(named, holders_of, facet_triangles, box_of,
                                 work.seeds, work.lift);
  trace_tet_crossings(points, triangles, faces, live, facet_of, work.seeds,
                      work);
  const auto &cut = work.cut;
  drain.cut = cut.size();
  if (!cut.size())
    return drain;

  state_tet_pair_records(points, triangles, cut, identity, band, true, work);
  const auto &records = work.records;
  const std::size_t standing = points.size();
  state_tet_point_keys<Index>(records, cut, parents, work.record_pairs);
  name_tet_record_points<Index, Int>(points, records, work.global,
                                     work.positions, work.record_pairs,
                                     authority, work.point_of, work.record_ids);
  state_tet_cut_contacts<Index>(
      records, cut, work.point_of, facet_of, edge_key, corners_of,
      [&authority](Index p) { return authority.standing(p); }, work.contacts,
      work.face_contacts, work.edge_points, work.cut_chords, work.ids);

  // a chord crossing is proposed before it is named: past `proposal_base`
  // an id is a proposal, named by exact position once the crossings stand
  auto &proposed = work.proposed;
  proposed.clear();
  const Index proposal_base = Index(points.size());
  const auto point = [&points, &proposed, proposal_base](Index p) -> pt_t {
    return p < proposal_base ? pt_t(points[std::size_t(p)])
                             : proposed[std::size_t(p - proposal_base)];
  };
  const auto propose = [&proposed, proposal_base](const pt_t &p) -> Index {
    proposed.push_back(p);
    return Index(proposal_base + Index(proposed.size()) - 1);
  };
  const auto axes_of = [&](Index f) {
    const auto c = corners_of(cut[std::size_t(f)]);
    return tf::exact::projection_axes<Int>(point(c[0]), point(c[1]),
                                           point(c[2]));
  };
  const auto parent_of = [&](Index f) {
    return Index(parents[std::size_t(cut[std::size_t(f)])]);
  };
  state_tet_chord_crossings<Index, Int>(work.cut_chords, point, axes_of,
                                        parent_of, propose, work.chord_points,
                                        work.crossings, work.chord_work);
  name_tet_proposals<Index, Int>(points, proposed, proposal_base, authority,
                                 work.crossings, work.chord_points,
                                 work.proposal_name, work.no_pairs);
  proposed.clear();
  // past the naming every id is a point of `points`
  const auto placed = [&points](Index p) -> pt_t {
    return pt_t(points[std::size_t(p)]);
  };
  land_tet_edge_chords<Int>(work.cut_chords, work.chord_points, cut, placed,
                            edge_key, work.edge_points, work.run_a);
  land_tet_reach_points<Int>(points, work.edge_points, work.holders, live,
                             holds, corners_of, edge_key,
                             Int(identity.quantum()), work.landing);
  // every side row at its own parameter, and each side's rows welded along
  // it, before the lanes read them
  const auto slot_point = [&](Index slot) {
    return placed(authority.identity_at(slot));
  };
  state_tet_side_parameters<Int>(work.edge_points, slot_point, placed,
                                 work.side_t);
  weld_tet_side_points<Int>(
      work.edge_points, work.side_t, slot_point,
      [&authority](Index slot) { return authority.identity_at(slot); },
      Int(identity.quantum()), authority, work.side_order);

  const std::size_t n_groups = state_tet_lanes<Index>(
      cut, work.contacts, work.edge_points, work.holders, live, holds,
      work.unit_runs, work.unit_members, work.unit_of_cut, work.neighbour_units,
      work.due_units, work.group, work.grouped, work.touched);
  const auto &edge_points = work.edge_points;
  const auto points_on_edge = [&](Index t, std::size_t k, auto &&visit) {
    const auto key = edge_key(t, k);
    for (auto at =
             std::lower_bound(edge_points.begin(), edge_points.end(),
                              std::array<Index, 3>{key[0], key[1], Index(-1)});
         at != edge_points.end() && (*at)[0] == key[0] && (*at)[1] == key[1];
         ++at)
      visit((*at)[2]);
  };
  const auto rows_on_edge = [&](Index t, std::size_t k, auto &&visit) {
    const auto key = edge_key(t, k);
    for (auto at =
             std::lower_bound(edge_points.begin(), edge_points.end(),
                              std::array<Index, 3>{key[0], key[1], Index(-1)});
         at != edge_points.end() && (*at)[0] == key[0] && (*at)[1] == key[1];
         ++at)
      visit(std::size_t(at - edge_points.begin()));
  };
  drain.failed = wave_tet_lanes<Index, Int>(
      points, Int(identity.quantum()), work.due_units, n_groups, work.unit_runs,
      work.unit_members, work.neighbour_units, cut, corners_of, edge_key,
      rows_on_edge, work.edge_points, work.side_t, slot_point, work.contacts,
      work.face_contacts, work.cut_chords, work.chord_points, placed,
      work.holders, live, holds, work.changed, work.all_pieces,
      work.stated_sides, authority, work.wave);
  work.edge_points.reserve(work.edge_points.size() + work.stated_sides.size());
  for (const auto &row : work.stated_sides)
    work.edge_points.push_back(row);
  std::sort(work.edge_points.begin(), work.edge_points.end());
  work.edge_points.erase_till_end(
      std::unique(work.edge_points.begin(), work.edge_points.end()));
  const auto speaks = [&authority](Index p) { return authority.standing(p); };
  enter_tet_retired_holders<Index>(
      work.wave.member_triangle, faces, corners_of, work.holders, live, speaks,
      work.changed, work.all_pieces, work.retired_slots, work.entered);

  drain.delta = cut.size() + work.neighbour_units.size();
  drain.created = points.size() - standing;
  if (!drain.failed)
    drain.failed = count_tet_drain_t_junctions(
        edge_points, work.holders, live, holds, corners_of, edge_key, speaks,
        work.changed, work.all_pieces);
  // a member whose pieces do not run its own boundary once is one the
  // lattice cannot hold apart there: the drain is refused
  if (!drain.failed)
    drain.failed = count_tet_drain_member_openness<Int>(
        work.changed, work.all_pieces, corners_of, points_on_edge, placed,
        speaks);
  if (drain.failed)
    return drain;
  restate_tet_cut_facets(
      work.changed, work.all_pieces, cut, facet_of, facet_offsets, parents,
      triangles, corners_of,
      [&authority](Index v) { return authority.seat_of(v) >= Index(0); },
      retired, rows, sources, beside, work.cut_facets, drain);
  return drain;
}

} // namespace tf::tetmesh
