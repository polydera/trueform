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
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

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
          typename VertexOffsets>
auto state_plane_refusal(
    const Groups &groups, Index plane, const Cdt &cdt,
    const tf::buffer<Index> &ends, const tf::buffer<Index> &constraint_groups,
    Index n_flat, const VertexOffsets &vertex_offsets, Index crossing_kind,
    Param whole,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology,
    std::size_t &landings, std::size_t &crossings) -> void {
  const auto kept = cdt.index_map().kept_ids();
  const auto name_frame_t = [&](Index ticket, Param t) {
    return plane_recovery_flat_edge_inverted(plane_recovery_flat_edge_ends(
               groups.canon_group(ticket)[0], n_flat, vertex_offsets))
               ? Param(whole - t)
               : t;
  };
  for (const auto &crossing : cdt.parameterized_crossings()) {
    if (crossing.id_b == Index(-1)) {
      const auto input = kept[std::size_t(crossing.point)];
      if (input == Index(ends.size()))
        continue;
      const auto flat = ends[std::size_t(input)];
      const auto feature =
          flat < n_flat
              ? tf::intersect::graph::plane_vertex_name<Index>(flat)
              : tf::intersect::graph::plane_point_name<Index>(flat - n_flat);
      const auto name = make_plane_recovery_name<Index, Param>(feature);
      const auto ticket = constraint_groups[std::size_t(crossing.id_a)];
      const auto root = groups.group_of(ticket);
      const auto tier = groups.tier_of(ticket);
      statements.push_back({name, root, plane, tier});
      topology.push_back({name, root, plane, tier,
                          name_frame_t(ticket, crossing.t_a),
                          std::uint8_t(1)});
      ++landings;
      continue;
    }
    const auto group_a = constraint_groups[std::size_t(crossing.id_a)];
    const auto group_b = constraint_groups[std::size_t(crossing.id_b)];
    if (group_a == group_b)
      continue;
    Index birth_root = Index(-1), birth_partner = Index(-1);
    const auto feature = make_plane_recovery_crossing_name(
        groups, group_a, group_b, n_flat, vertex_offsets, crossing_kind,
        birth_root, birth_partner);
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

} // namespace tf::arrangement
