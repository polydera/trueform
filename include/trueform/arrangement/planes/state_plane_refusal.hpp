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
#include "../../intersect/graph/plane_identity_names.hpp"
#include "./find_plane_meeting_landing.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// A parameter along the constraint `ticket` as the plane fed it, in the frame
/// of the name its row belongs to (THE FRAME LAW below).
template <typename Index, typename Param, typename Groups,
          typename VertexOffsets>
auto plane_name_frame_parameter(const Groups &groups, Index ticket, Param t,
                                Index n_flat,
                                const VertexOffsets &vertex_offsets,
                                Param whole) -> Param {
  return plane_recovery_flat_edge_inverted(plane_recovery_flat_edge_ends(
             groups.canon_group(ticket)[0], n_flat, vertex_offsets))
             ? Param(whole - t)
             : t;
}

/// One landing, whatever saw it: the identity `name` lands on the constraint
/// `ticket`, at `t` along it as the plane fed it. A landing on an identity
/// that already stands proposes its birth; one named by the meeting that
/// minted its point does not, the meeting's own carriers do.
template <typename Index, typename Param, typename Groups, typename Name,
          typename VertexOffsets>
auto state_plane_landing(
    const Groups &groups, Index plane, const Name &name, Index ticket, Param t,
    bool birth, Index n_flat, const VertexOffsets &vertex_offsets, Param whole,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology) -> void {
  const auto root = groups.group_of(ticket);
  const auto tier = groups.tier_of(ticket);
  statements.push_back({name, root, plane, tier});
  topology.push_back(
      {name, root, plane, tier,
       plane_name_frame_parameter(groups, ticket, t, n_flat, vertex_offsets,
                                  whole),
       std::uint8_t(birth ? 1 : 0)});
}

/// The name of the identity a flat id stands for.
template <typename Index, typename Param>
auto plane_flat_identity_name(Index flat, Index n_flat) {
  return make_plane_recovery_name<Index, Param>(
      flat < n_flat
          ? tf::intersect::graph::plane_vertex_name<Index>(flat)
          : tf::intersect::graph::plane_point_name<Index>(flat - n_flat));
}

/// State what a plane that refused its triangulation still saw.
///
/// The resolve-mode triangulation had to create every junction it reports, and
/// each one names the exact pieces it joins — a landing the identity it lands
/// on, a crossing the two pieces themselves — with the dyadic parameter that
/// places it on each. An edge is an edge: every constraint arrives as its
/// TICKET, and the ticket carries its own tier — which table indexes that id
/// — so a carrier reading both tiers splits each constraint where it lives.
///
/// `statements` and `topology` are stated in ONE breath, one proposal per
/// statement, so the two carriers close into the same classes and their
/// correspondence is position. Each plane states its local birth candidate;
/// @ref tf::arrangement::close_plane_recovery_births elects once after equal
/// names from independent planes meet.
///
/// THE FRAME LAW: the triangulation measures a junction along the constraint
/// as the plane fed it — the definition's own order — while the row it fills
/// belongs to a NAME, whose frame is that definition's endpoint flats
/// ascending. The two disagree on a piece with one created and one original
/// endpoint, so every parameter is stated in the name's frame here, at
/// emission, where the row and its edge are one fact.
template <typename Index, typename Param, typename Groups, typename Cdt,
          typename Points2, typename VertexOffsets>
auto state_plane_refusal(
    const Groups &groups, Index plane, const Cdt &cdt,
    const tf::buffer<Index> &ends, const tf::buffer<Index> &cons,
    const Points2 &pts2, const tf::buffer<Index> &constraint_groups,
    Index n_flat, Index first_minted, const VertexOffsets &vertex_offsets,
    Index crossing_kind,
    Param whole,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology,
    tf::buffer<Index> &meeting_record, std::size_t &landings,
    std::size_t &crossings) -> void {
  const auto kept = cdt.index_map().kept_ids();
  const auto &records = cdt.parameterized_crossings();
  const auto meeting_landing = [&](const auto &crossing) {
    return find_plane_meeting_landing<Index, Param>(
        cons, pts2, ends, first_minted, crossing.id_a, crossing.id_b);
  };
  const auto name_frame_t = [&](Index ticket, Param t) {
    return plane_name_frame_parameter(groups, ticket, t, n_flat,
                                      vertex_offsets, whole);
  };
  const auto crossing_name = [&](const auto &crossing, Index &birth_root,
                                 Index &birth_partner) {
    return make_plane_recovery_crossing_name(
        groups, constraint_groups[std::size_t(crossing.id_a)],
        constraint_groups[std::size_t(crossing.id_b)], n_flat, vertex_offsets,
        crossing_kind, birth_root, birth_partner);
  };
  const auto sides_of_one_face = [&](Index x, Index y) {
    for (const auto &a : groups.canon_group(x))
      if (a.side >= 0)
        for (const auto &b : groups.canon_group(y))
          if (b.side >= 0 && a.face == b.face)
            return true;
    return false;
  };
  // A created point is named by the meeting that minted it, the first record
  // stating it; a constraint landing on it joins that name.
  meeting_record.allocate(std::size_t(cdt.points().size()));
  std::fill(meeting_record.begin(), meeting_record.end(), Index(-1));
  for (std::size_t at = records.size(); at-- != 0;)
    if (records[at].id_b != Index(-1))
      meeting_record[std::size_t(records[at].point)] = Index(at);
  for (const auto &crossing : records) {
    if (crossing.id_b == Index(-1)) {
      const auto input = kept[std::size_t(crossing.point)];
      const auto ticket = constraint_groups[std::size_t(crossing.id_a)];
      if (input == Index(ends.size())) {
        const auto at = meeting_record[std::size_t(crossing.point)];
        if (at == Index(-1))
          continue;
        const auto &meeting = records[std::size_t(at)];
        const auto group_a = constraint_groups[std::size_t(meeting.id_a)];
        const auto group_b = constraint_groups[std::size_t(meeting.id_b)];
        if (group_a == group_b || ticket == group_a || ticket == group_b)
          continue;
        // a meeting stated as a landing, or as nothing, mints no point
        const auto stated = meeting_landing(meeting);
        if (stated.shared || stated.point != Index(-1))
          continue;
        // a side of a face is never landed on a meeting another side of the
        // same face carries. Two sides meeting at a corner would share the
        // meeting beside it and fold the face onto itself between — on a
        // triangle that is every pair; the rule withholds the landing for
        // any two sides of one face, adjacent or not
        if (sides_of_one_face(ticket, group_a) ||
            sides_of_one_face(ticket, group_b))
          continue;
        Index birth_root = Index(-1), birth_partner = Index(-1);
        const auto feature = crossing_name(meeting, birth_root, birth_partner);
        if (feature[1] == feature[3] && feature[2] == feature[4])
          continue;
        state_plane_landing(groups, plane,
                            make_plane_recovery_name<Index, Param>(feature),
                            ticket, crossing.t_a, false, n_flat,
                            vertex_offsets, whole, statements, topology);
        ++landings;
        continue;
      }
      state_plane_landing(groups, plane,
                          plane_flat_identity_name<Index, Param>(
                              ends[std::size_t(input)], n_flat),
                          ticket, crossing.t_a, true, n_flat, vertex_offsets,
                          whole, statements, topology);
      ++landings;
      continue;
    }
    const auto group_a = constraint_groups[std::size_t(crossing.id_a)];
    const auto group_b = constraint_groups[std::size_t(crossing.id_b)];
    if (group_a == group_b)
      continue;
    const auto landing = meeting_landing(crossing);
    if (landing.shared)
      continue;
    if (landing.point != Index(-1)) {
      state_plane_landing(groups, plane,
                          plane_flat_identity_name<Index, Param>(
                              ends[std::size_t(landing.point)], n_flat),
                          constraint_groups[std::size_t(landing.constraint)],
                          landing.parameter, true, n_flat, vertex_offsets,
                          whole, statements, topology);
      ++landings;
      continue;
    }
    Index birth_root = Index(-1), birth_partner = Index(-1);
    const auto feature = crossing_name(crossing, birth_root, birth_partner);
    if (feature[1] == feature[3] && feature[2] == feature[4])
      continue;
    const auto name = make_plane_recovery_name<Index, Param>(feature);
    statements.push_back(
        {name, groups.group_of(group_a), plane, groups.tier_of(group_a)});
    statements.push_back(
        {name, groups.group_of(group_b), plane, groups.tier_of(group_b)});
    const bool a_born = birth_root == group_a;
    const auto birth_t = a_born ? crossing.t_a : crossing.t_b;
    const auto partner_t = a_born ? crossing.t_b : crossing.t_a;
    topology.push_back({name, groups.group_of(birth_root), plane,
                        groups.tier_of(birth_root),
                        name_frame_t(birth_root, birth_t), std::uint8_t(1)});
    topology.push_back({name, groups.group_of(birth_partner), plane,
                        groups.tier_of(birth_partner),
                        name_frame_t(birth_partner, partner_t),
                        std::uint8_t(0)});
    ++crossings;
  }
}

/// State the landings the rim law found, in the shape a triangulation's own
/// landing takes: the identity that lands names the statement, and the group
/// of the constraint it lands on carries it at the landing's parameter in the
/// name's frame.
template <typename Index, typename Param, typename Groups,
          typename VertexOffsets, typename Landings>
auto state_plane_rim_landings(
    const Groups &groups, Index plane, const Landings &landings,
    const tf::buffer<Index> &ends, const tf::buffer<Index> &constraint_groups,
    Index n_flat, const VertexOffsets &vertex_offsets, Param whole,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology,
    std::size_t &rim_landings) -> void {
  for (const auto &landing : landings) {
    state_plane_landing(groups, plane,
                        plane_flat_identity_name<Index, Param>(
                            ends[std::size_t(landing.point)], n_flat),
                        constraint_groups[std::size_t(landing.constraint)],
                        landing.parameter, true, n_flat, vertex_offsets, whole,
                        statements, topology);
    ++rim_landings;
  }
}

} // namespace tf::arrangement
