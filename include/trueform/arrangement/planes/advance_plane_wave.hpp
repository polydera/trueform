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
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./canonicalize_plane_wave.hpp"
#include "./close_plane_recovery_births.hpp"
#include "./close_plane_recovery_identities.hpp"
#include "./close_plane_recovery_names.hpp"
#include "./commit_plane_wave_blocks.hpp"
#include "./map_plane_split_parents.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_diff.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_round_evidence.hpp"
#include "./port_plane_diff.hpp"
#include "./promote_plane_collision_carriers.hpp"
#include "./propose_plane_split_pieces.hpp"
#include "./weld_local_plane_rows.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// What one wave did with the round's evidence: it stated new splits, it
/// committed identities without changing a constraint set, it consumed
/// nothing, it named a fact this arrangement's producers cannot publish, or
/// it found that what it is about to state reaches a face the cut world never
/// named and handed the round back unconsumed.
enum class plane_wave_result {
  stated,
  committed,
  exhausted,
  unsupported,
  entrance
};

/// WAVE. Consume one round's evidence: close the statements and the welds into
/// identities, order the splits they place on the exact pieces they name,
/// move every carrier they touch from the world tier into this
/// arrangement's own, substitute the merged identities into the rows that
/// name them, subdivide the split parents, close everything under equal key,
/// and publish as the next round's frontier the carriers whose CONSTRAINT
/// SET changed. A carrier whose rows only changed identity is not among
/// them: it keeps the triangulation it already has, which is also what makes
/// a weld wave terminate — it cannot restate itself. A carrier that REFUSED
/// keeps nothing, so a substitution reaching its rows does publish it, and
/// the wave still terminates: rejoining costs a fresh retirement.
///
/// A wave that states no live split and retires no identity consumed nothing
/// and ends the loop, and so does one whose statement the local tier cannot
/// express: those refusals leave the tier exactly as they found it, so the
/// planes that refused keep the failure this round published for them and
/// every other plane keeps its product. `unsupported` is kept for a fact
/// this arrangement's own producers own — a statement/proposal pairing that
/// cannot be, a class the identity election did not name, or a tier the
/// diff, the port or the substitution itself refuses — where publishing
/// would publish an identity nothing proved. Once the substitution has
/// published, every remaining seam answers the same way: the tier states
/// identities whose closure the wave owes it.
///
/// THE ENTRANCE, when a caller asks for it, and it is the SAME LAW BOTH
/// TIERS OBEY: a split this wave is about to state on a group still the
/// WORLD'S, landing on an ORIGINAL SIDE, reaches every face holding that
/// edge, and an ORIGINAL this wave retires reaches every face incident to
/// it — while a group still the world's holds only the faces the cut world
/// named. The wave publishes both sets and hands the round back UNCONSUMED.
/// Nothing this arrangement owns has been written at that point — the
/// identities, the merges and the splits are all still the round's own
/// locals — so the evidence is DISCARDED, not translated: it is seen again
/// next round, against the promoted tables.
///
/// The wave's identity and census facts are the wave's until every seam of
/// it validates; they join this arrangement in one place, last.
template <typename Index, typename Int, typename World, typename VertexOffsets,
          typename ExactPoint, typename NamePoint,
          typename StateEntrance = tf::none_t>
auto advance_plane_wave(
    const World &world, plane_round_evidence<Index, Int> &evidence,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router,
    tf::buffer<std::array<Index, 3>> &merges, tf::buffer<Index> &created_class,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &class_t,
    tf::buffer<plane_recovery_name<
        Index, typename tf::exact::meta<Int>::param_type>> &class_name,
    const VertexOffsets &vertex_offsets, Index n_flat, Index base_created,
    const ExactPoint &exact_point, const NamePoint &name_point,
    plane_arrangement_census &census, tf::buffer<Index> &frontier,
    const StateEntrance &state_entrance = tf::none) -> plane_wave_result {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  auto &delta = evidence.statements;
  auto &topology = evidence.topology;
  if (delta.size() == 0 && evidence.welds.size() == 0)
    return plane_wave_result::exhausted;
  tf::buffer<Index> topology_offsets;
  close_plane_recovery_proposals(topology, topology_offsets);
  tf::buffer<Index> delta_offsets;
  close_plane_recovery_names<Index, param_t>(delta, delta_offsets);

  tf::buffer<Index> delta_class_birth;
  tf::buffer<param_t> delta_class_t;
  if (!close_plane_recovery_births(delta, delta_offsets, topology,
                                   topology_offsets, delta_class_birth))
    return plane_wave_result::unsupported;
  // every class a piece can carry survived; a wave left with none states
  // nothing, and its planes keep the failure they were published with
  if (delta.size() == 0 && evidence.welds.size() == 0)
    return plane_wave_result::exhausted;
  const auto n_delta = delta_offsets.size() == 0
                           ? Index(0)
                           : Index(delta_offsets.size()) - Index(1);
  delta_class_t.allocate(std::size_t(n_delta));
  tf::parallel_transform(
      tf::make_range(delta_class_birth), delta_class_t,
      [&](Index birth) { return topology[std::size_t(birth)].parameter; },
      tf::checked);

  // THE CANDIDATE ENDPOINTS a new name may land on. The births barrier
  // accepted only because EVERY proposal names a root, so the candidate set
  // is the proposals' own edges; they are deduped as the identities they
  // are, in the flat vertex currency, never as the edges that mention them.
  //
  // The undeduped table stays: `topology_offsets` groups the proposals by
  // class, so it also states, per class, the endpoints of the constraints
  // THAT class splits — which is what a landing is decided against.
  tf::buffer<Index> proposal_ends;
  proposal_ends.allocate(topology.size() * 2);
  tf::parallel_for_each(
      tf::make_sequence_range(topology.size()),
      [&](std::size_t at) {
        const auto &proposal = topology[at];
        const auto ends = plane_recovery_flat_edge_ends(
            proposal.tier == Index(0)
                ? world.canon_group(proposal.root)[0]
                : local_tables.canon_group(proposal.root)[0],
            n_flat, vertex_offsets);
        proposal_ends[at * 2] = ends[0];
        proposal_ends[at * 2 + 1] = ends[1];
      },
      tf::checked);
  tf::buffer<Index> candidate_points;
  candidate_points.allocate(proposal_ends.size());
  tf::parallel_copy(proposal_ends, candidate_points);
  tbb::parallel_sort(candidate_points.begin(), candidate_points.end());
  candidate_points.erase_till_end(
      std::unique(candidate_points.begin(), candidate_points.end()));

  tf::buffer<Index> delta_class_id;
  delta_class_id.allocate(std::size_t(n_delta));
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_delta)),
      [&](std::size_t cls) {
        const auto &feature =
            delta[std::size_t(delta_offsets[cls])].name.feature;
        delta_class_id[cls] = feature[0] == Index(0) ? feature[1] : Index(-1);
      },
      tf::checked);
  const auto class_base = Index(class_t.size());
  const auto created_base = Index(created_class.size());
  // The created extent BEFORE this wave's own mints, stated once. A cut at
  // or above it is an identity this wave minted, and a fresh identity
  // appears in no standing row, so the canonicalize sweeps only for the
  // cuts below it: too high costs sweep work, too low MISSES a standing
  // collision.
  const auto created_mint_base = base_created + created_base;
  tf::buffer<Index> delta_created_class;
  tf::buffer<std::array<Index, 3>> delta_merges;
  const auto class_point = [&](Index cls) {
    return name_point(delta_class_t[std::size_t(cls)],
                      delta[std::size_t(delta_offsets[std::size_t(cls)])].name);
  };
  close_plane_recovery_identities(
      exact_point, class_point, vertex_offsets, delta, delta_offsets, n_delta,
      base_created, n_flat, delta_class_id, delta_created_class,
      evidence.welds, merges, delta_merges, candidate_points, topology_offsets,
      proposal_ends, created_class, created_base);

  // A proposal reads the identity of its OWN class: the two carriers were
  // stated in one breath and closed on one key, so class position is the
  // join, and an unnamed class is an election that did not finish.
  if (tf::parallel_contains(
          tf::make_sequence_range(n_delta),
          [&delta_class_id](Index cls) {
            return delta_class_id[std::size_t(cls)] == Index(-1);
          },
          tf::checked))
    return plane_wave_result::unsupported;

  // THE WAVE'S TABLE: the standing rewrites plus its own, closed once, so
  // one search answers what any identity of this arrangement speaks now.
  tf::buffer<std::array<Index, 3>> wave_merges = merges;
  tf::core::append(delta_merges, wave_merges);
  tf::intersect::graph::close_plane_merges<Index>(wave_merges);
  // THE RETIRED DELTA — the wave's own changed set, flat and ascending by
  // construction: the rows are sorted by source inside each kind, and the
  // created kind lifts past the whole original space.
  tf::buffer<Index> retired;
  retired.allocate(delta_merges.size());
  tf::parallel_transform(
      tf::make_range(delta_merges), retired,
      [&](const std::array<Index, 3> &row) {
        return row[0] == Index(0)
                   ? row[1]
                   : tf::intersect::graph::flat_of_vertex(
                         vertex_offsets, std::int16_t(-1), row[1]);
      },
      tf::checked);

  tf::buffer<Index> split_edge;
  tf::buffer<Index> split_tier;
  tf::buffer<Index> split_offsets;
  tf::buffer<Index> split_data;
  std::size_t out_of_span = 0;
  std::size_t on_endpoint = 0;
  order_plane_recovery_splits(
      world, local_tables, topology, topology_offsets, delta_class_id,
      wave_merges, vertex_offsets, n_delta,
      param_t(1) << tf::exact::meta<Int>::param_bits, true, split_edge,
      split_tier, split_offsets, split_data, out_of_span, on_endpoint);
  if (split_edge.size() == 0 && retired.size() == 0)
    return plane_wave_result::exhausted;

  // THE ENTRANCE'S TRIGGER, stated where the splits are ordered and the
  // pieces are not yet proposed: the roots still the WORLD'S that cut an
  // ORIGINAL SIDE, and the ORIGINALS this wave retires. A world group holds
  // only the faces the cut world named, so such a split reaches faces that
  // own no row anywhere, and a retired original reaches its whole ring; the
  // caller that can promote them is handed the round back untouched. The
  // rows are few — one per split edge — so the walk is its own grain.
  if constexpr (!std::is_same<StateEntrance, tf::none_t>::value) {
    tf::buffer<Index> entrance_roots;
    for (std::size_t at = 0; at != split_edge.size(); ++at) {
      if (split_tier[at] != Index(0))
        continue;
      for (const auto &instance : world.canon_group(split_edge[at]))
        if (instance.side >= 0) {
          entrance_roots.push_back(split_edge[at]);
          break;
        }
    }
    const auto entrance_originals =
        tf::intersect::graph::plane_retired_originals(delta_merges);
    if ((entrance_roots.size() != 0 || entrance_originals.size() != 0) &&
        state_entrance(tf::make_range(entrance_roots), entrance_originals))
      return plane_wave_result::entrance;
  }

  tf::buffer<Index> weld_row_offsets;
  tf::buffer<Index> weld_row_data;
  const auto plane_of_face = [&world](Index face) {
    return world.plane_of_face(face);
  };
  // THE COLLISION PROBE, built once for the wave: a key this wave states
  // can equal a standing one only through a PRE-EXISTING identity — a cut
  // or a merge target below the mint base — and an identity the wave
  // itself mints appears in no standing row.
  tf::buffer<Index> standing_probe;
  tf::generic_generate(
      tf::make_range(split_data), standing_probe,
      [created_mint_base](Index cut, tf::buffer<Index> &out) {
        if (cut < created_mint_base)
          out.push_back(cut);
      },
      tf::checked);
  tf::generic_generate(
      tf::make_range(retired), standing_probe,
      [&](Index flat, tf::buffer<Index> &out) {
        const auto target = tf::intersect::graph::plane_merge_target(
            wave_merges, flat < n_flat ? Index(0) : Index(1),
            flat < n_flat ? flat : flat - n_flat);
        if (target < created_mint_base)
          out.push_back(target);
      },
      tf::checked);
  if (standing_probe.size() != 0) {
    tbb::parallel_sort(standing_probe.begin(), standing_probe.end());
    standing_probe.erase_till_end(
        std::unique(standing_probe.begin(), standing_probe.end()));
  }
  tf::buffer<Index> taken;
  if (!gather_plane_diff(world, local_tables, plane_ticket, group_router,
                         retired, vertex_offsets, split_edge, split_tier,
                         plane_of_face, weld_row_offsets, weld_row_data,
                         frontier, taken) ||
      (split_edge.size() != 0 && frontier.size() == 0))
    return plane_wave_result::unsupported;
  // THE OWNERSHIP LAW, and the whole reason the frontier is a RING: a group
  // whose statement CHANGES goes local and takes every carrier of it in the
  // same wave. AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH
  // CARRIERS, so it drags nobody, and a plane the wave never touched keeps
  // reading — and publishing — exactly what it always did.
  promote_plane_collision_carriers(world.tables(), local_tables, plane_ticket,
                                   standing_probe, plane_of_face, frontier,
                                   taken);
  if (taken.size() != 0) {
    tbb::parallel_sort(taken.begin(), taken.end());
    taken.erase_till_end(std::unique(taken.begin(), taken.end()));
  }

  if (frontier.size() != 0) {
    tf::buffer<Index> recdt_planes;
    tf::buffer<Index> changed_groups;
    if (!port_plane_diff(world.tables(), tf::make_range(frontier), taken,
                         local_tables, plane_ticket, group_router) ||
        !weld_local_plane_rows(world.tables(), wave_merges, vertex_offsets,
                               weld_row_offsets, weld_row_data, plane_of_face,
                               local_tables, plane_ticket, group_router,
                               recdt_planes, changed_groups))
      return plane_wave_result::unsupported;
    // The substitution has published, and its survivors hold keys only the
    // canonicalize can close: from here a refusal is an identity state this
    // arrangement cannot publish, not a wave it can decline.
    const auto incomplete =
        changed_groups.size() == 0 && recdt_planes.size() == 0
            ? plane_wave_result::exhausted
            : plane_wave_result::unsupported;

    tf::buffer<Index> split_parents;
    tf::buffer<Index> local_split_offsets;
    tf::buffer<Index> local_split_data;
    tf::buffer<plane_split_piece_layout<Index>> piece_layout;
    tf::buffer<def_t> piece_proposals;
    tf::buffer<Index> carrier_planes;
    tf::buffer<Index> final_row;
    tf::buffer<std::array<Index, 2>> losers;
    // Propose, fuse, commit: the children are minted only after equal keys
    // have met, so no unfused state exists to detect. A loser's carrier the
    // rebuild did not touch is substituted in place, which is why the
    // frontier the diff published is the whole frontier.
    if (!map_plane_split_parents(split_edge, split_tier, split_offsets,
                                 split_data, group_router, split_parents,
                                 local_split_offsets, local_split_data) ||
        !propose_plane_split_pieces(split_parents, local_split_offsets,
                                    local_split_data, plane_of_face,
                                    local_tables, plane_ticket, piece_layout,
                                    piece_proposals, carrier_planes))
      return incomplete;
    // The composed transaction preflights every later fallible extent BEFORE
    // the canonicalize publishes: a commit refusal after a successful fuse
    // would leave retired roots beside standing carrier blocks. The bound is
    // conservative — every carrier block rewritten whole plus every proposal
    // — so a pass here makes the commit's own exact guards unreachable.
    {
      const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
      const auto rewritten_rows =
          piece_proposals.size() +
          tf::reduce(tf::make_mapped_range(
                         tf::make_range(carrier_planes),
                         [&](Index plane) {
                           const auto block = plane_ticket[std::size_t(plane)];
                           return tf::make_range(
                                      local_tables.edges())[std::size_t(block)]
                               .size();
                         }),
                     [](std::size_t x, std::size_t y) { return x + y; },
                     std::size_t(0), tf::checked);
      if (carrier_planes.size() > index_extent - local_tables.edges().size() ||
          rewritten_rows >
              index_extent - local_tables.edges().data_buffer().size())
        return incomplete;
    }
    if (!canonicalize_plane_wave(world.tables(), piece_layout, piece_proposals,
                                 standing_probe, plane_ticket, plane_of_face,
                                 local_tables, group_router, final_row, losers,
                                 changed_groups))
      return incomplete;
    if (!commit_plane_wave_blocks(world.tables(), piece_layout, final_row,
                                  carrier_planes, losers, plane_of_face,
                                  local_tables, plane_ticket))
      return incomplete;

    // A REFUSED plane keeps no triangulation, so the rows the substitution
    // rewrote are a constraint set it has never read: the weld's "a survivor
    // keeps its product" leaves it with none, and the coincidence the merge
    // retired may be the very collision it refused on.
    tf::buffer<Index> welded_refusals;
    if (weld_row_offsets.size() != 0)
      tf::generic_generate(
          tf::make_range(evidence.refused), welded_refusals,
          [&weld_row_offsets](Index plane, tf::buffer<Index> &out) {
            if (weld_row_offsets[std::size_t(plane) + 1] >
                weld_row_offsets[std::size_t(plane)])
              out.push_back(plane);
          },
          tf::checked);

    // THE ROUND'S FRONTIER is the carriers whose constraint set changed: the
    // split parents', the ones a retirement rebuilt, and the refusals a
    // substitution reached. A surviving weld changed identities alone, so its
    // carriers keep their triangulations.
    if (recdt_planes.size() == 0 && welded_refusals.size() == 0)
      frontier = std::move(carrier_planes);
    else {
      frontier = std::move(recdt_planes);
      tf::core::append(welded_refusals, frontier);
      tf::core::append(carrier_planes, frontier);
      tbb::parallel_sort(frontier.begin(), frontier.end());
      frontier.erase_till_end(std::unique(frontier.begin(), frontier.end()));
    }
  }

  const auto n_delta_classes = delta_offsets.size() == 0
                                   ? Index(0)
                                   : Index(delta_offsets.size()) - Index(1);
  delta_class_t.reallocate_and_initialize(std::size_t(n_delta_classes),
                                          param_t(0));
  class_t.reallocate(std::size_t(class_base) + std::size_t(n_delta_classes));
  class_name.reallocate(std::size_t(class_base) + std::size_t(n_delta_classes));
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_delta_classes)),
      [&](std::size_t cls) {
        class_t[std::size_t(class_base) + cls] = delta_class_t[cls];
        class_name[std::size_t(class_base) + cls] =
            delta[std::size_t(delta_offsets[cls])].name;
      },
      tf::checked);
  const auto created_write = created_class.size();
  created_class.reallocate(created_write + delta_created_class.size());
  tf::parallel_for_each(
      tf::make_sequence_range(delta_created_class.size()),
      [&](std::size_t at) {
        created_class[created_write + at] =
            delta_created_class[at] + class_base;
      },
      tf::checked);
  merges = std::move(wave_merges);
  census.splits += split_data.size();
  census.splits_out_of_span += out_of_span;
  census.splits_on_endpoint += on_endpoint;
  return frontier.size() == 0 ? plane_wave_result::committed
                              : plane_wave_result::stated;
}

} // namespace tf::arrangement
