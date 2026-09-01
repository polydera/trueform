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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_piece_definition_instance.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"
#include "./promote_plane_collision_carriers.hpp"
#include "./propose_plane_split_pieces.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <tuple>

namespace tf::arrangement {

/// Close the wave's proposals under EQUAL KEY, and mint the survivors.
///
/// One key is one identity, so every proposal, and every standing group a
/// proposal collides with, is one class of one sort. A class mints ONE fresh
/// group carrying the UNION of its definition instances: once-minted-forever
/// says an unchanged group keeps its id, and a fusion is a new identity.
/// Every loser retires by the shelf it lives on — a standing LOCAL group
/// structurally, because the blocks stop naming it, an immutable root through
/// the router, because the world tier is read-only forever.
///
/// THE COLLISION COSTS ONE SWEEP. A key this wave states can equal a standing
/// key only through a PRE-EXISTING identity — a cut or a merge target this
/// wave minted appears in no standing row — so those identities are the whole
/// probe: one flat pass at PLANE grain over the block each plane's ticket
/// names — the world tier while the ticket is `-1`, its local block otherwise
/// — collecting the groups whose rows name a probed identity. Only the changed
/// set is ever searched, and only its keys are matched against the wave's.
///
/// THE OWNERSHIP LAW makes every standing match a LOCAL one: a fusion CHANGES
/// a group, so every group a probed identity can reach was TAKEN with all its
/// carriers before the surgery ran, and every carrier's block was repointed
/// with it. A standing row still reading the world is therefore a state the
/// arrangement cannot be in, and the call rejects it. Nothing global is ever
/// written; the world tier is read-only storage whose logically retired
/// groups the router names.
///
/// `changed_groups` is the weld surgery's survivors, and they are PARTICIPANTS
/// of the same kind as a matched standing local group: the surgery restated
/// their rows in place, so their instances must survive into whatever their
/// new key closes into, and they enter as the standing rows they are. The
/// sweep may name them again — a changed key holds a merge target — and the
/// participants are one set, so a group entering twice enters once.
///
/// A class holding ONE changed group's instances and nothing else closes
/// nothing: that group already changed in place, and once-minted-forever mints
/// only for a fusion, so it keeps its identity and its class mints no group at
/// all.
///
/// GROUPS FUSE; INSTANCES NEVER DO. A minted span is the union of what every
/// contributor stated: the instance's whole identity is its provenance AND
/// its flags — `A->B` and `B->A` on one key are two facts, not one — so the
/// union deduplicates only TRUE repeats, and a child dominates the standing
/// row it fully restates: the child's flags are the piece maker's statement
/// about this wave. Two CHILDREN can state one instance too — two parents
/// carried by one face state the same piece the moment they coincide — and
/// the same law answers: one row survives, and every slot that stated it
/// reads that row.
///
/// `final_row` answers the proposal slots — one final definition row each,
/// the whole correspondence the block commit needs — and `losers` names every
/// structurally retired local group with the winner that supersedes it. The
/// router COMPOSES through the losers: an immutable root whose route names a
/// loser follows it to the winner, so a root always names its current local
/// descendant.
///
/// A standing group retired by this wave does not participate as standing
/// state: its children already restate its complete definition span. Thus a
/// child of one parent can close directly with a child of another parent even
/// when its key also equals that other's pre-wave parent key.
///
/// False rejects the call and publishes nothing: the tiers' interiors were
/// proven by the producers that appended them, so this states only what the
/// closure itself indexes — including a standing row on a still-immutable
/// plane and a child that did not actually shorten its own parent.
template <typename Index, typename Int, typename PlaneOfFace>
auto canonicalize_plane_wave(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::buffer<plane_split_piece_layout<Index>> &layout,
    const tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &proposals,
    const tf::buffer<Index> &standing_probe,
    const tf::buffer<Index> &plane_ticket,
    const PlaneOfFace &plane_of_face,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &group_router, tf::buffer<Index> &final_row,
    tf::buffer<std::array<Index, 2>> &losers,
    const tf::buffer<Index> &changed_groups) -> bool {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using key_t = std::array<Index, 4>;
  const auto n_parents = layout.size();
  const auto n_canon = local_tables.n_canon();
  const auto n_immutable = world_tables.n_canon();
  const auto n_local_blocks = Index(local_tables.edges().size());
  const auto n_world_planes = Index(world_tables.edges().size());
  if (plane_ticket.size() >
      std::size_t(std::numeric_limits<Index>::max()))
    return false;
  const auto n_planes = Index(plane_ticket.size());
  if (!local_tables.well_formed())
    return false;
  if (n_parents == 0 && changed_groups.size() == 0) {
    if (proposals.size() != 0)
      return false;
    final_row.clear();
    losers.clear();
    return true;
  }
  if (group_router.size() != std::size_t(n_immutable))
    return false;
  if (!plane_ticket_well_formed(plane_ticket, n_world_planes, n_local_blocks))
    return false;
  auto &def_offsets = local_tables.def_offsets();
  if (tf::parallel_contains(
          tf::make_sequence_range(n_parents),
          [&](std::size_t at) {
            const auto &split = layout[at];
            return split.parent < Index(0) || split.parent >= n_canon ||
                   (at != 0 && layout[at - 1].parent >= split.parent) ||
                   split.pieces < Index(2) || split.instances < Index(1) ||
                   def_offsets[std::size_t(split.parent) + 1] -
                           def_offsets[std::size_t(split.parent)] !=
                       split.instances;
          },
          tf::checked))
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(changed_groups.size()),
          [&](std::size_t at) {
            const auto group = changed_groups[at];
            return group < Index(0) || group >= n_canon ||
                   (at != 0 && changed_groups[at - 1] >= group) ||
                   def_offsets[std::size_t(group) + 1] <=
                       def_offsets[std::size_t(group)];
          },
          tf::checked))
    return false;
  {
    std::size_t slots = 0;
    for (std::size_t at = 0; at < n_parents; ++at) {
      if (layout[at].first != Index(slots))
        return false;
      slots += std::size_t(layout[at].pieces) *
               std::size_t(layout[at].instances);
    }
    if (proposals.size() != slots)
      return false;
  }
  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  if (proposals.size() > index_extent)
    return false;

  using standing_row_t = std::array<Index, 2>;
  tf::buffer<standing_row_t> standing;
  if (standing_probe.size() != 0) {
    tf::generic_generate(
        tf::make_sequence_range(n_planes), standing,
        [&](Index plane, tf::buffer<standing_row_t> &out) {
          const auto source = find_plane_definition_source(
              world_tables, local_tables, plane_ticket, plane);
          const auto tier = make_plane_tier_definitions(
              world_tables, local_tables, !source.immutable);
          for (const auto row : source.tables.plane_edges(source.block)) {
            const auto &def = tier[std::size_t(row)];
            if (!probes_plane_identity(standing_probe, def.point_tag_0,
                                       def.point_0) &&
                !probes_plane_identity(standing_probe, def.point_tag_1,
                                       def.point_1))
              continue;
            out.push_back(
                tier.immutable(row)
                    ? standing_row_t{group_router[std::size_t(def.id)], def.id}
                    : standing_row_t{def.id, Index(-1)});
          }
        });
    if (standing.size() != 0) {
      tbb::parallel_sort(standing.begin(), standing.end());
      standing.erase_till_end(std::unique(standing.begin(), standing.end()));
    }
  }
  // THE OWNERSHIP LAW: a probed identity on a row still the world's means a
  // group the wave never took — a state the arrangement cannot be in, because
  // the promotion swept the same probe before the port and took every group
  // it named, with every carrier
  if (tf::parallel_contains(
          tf::make_range(standing),
          [](const standing_row_t &row) { return row[1] != Index(-1); },
          tf::checked))
    return false;

  struct standing_t {
    key_t key;
    Index group;
  };
  tf::buffer<standing_t> matched;
  // the wave's key table exists to be searched by the standing rows; with no
  // standing row there is nothing to match, and the table is not built
  if (standing.size() != 0) {
    tf::buffer<key_t> wave_keys;
    tf::generic_generate(
        tf::make_range(layout), wave_keys,
        [&](const plane_split_piece_layout<Index> &split,
            tf::buffer<key_t> &out) {
          for (Index rank = 0; rank < split.pieces; ++rank)
            out.push_back(plane_piece_key<Index>(
                proposals[std::size_t(split.first + rank * split.instances)]));
        },
        tf::checked);
    tf::generic_generate(
        tf::make_range(changed_groups), wave_keys,
        [&](Index group, tf::buffer<key_t> &out) {
          out.push_back(
              plane_piece_key<Index>(local_tables.canon_group(group)[0]));
        },
        tf::checked);
    tbb::parallel_sort(wave_keys.begin(), wave_keys.end());
    wave_keys.erase_till_end(std::unique(wave_keys.begin(), wave_keys.end()));
    tf::generic_generate(
        tf::make_range(standing), matched,
        [&](const standing_row_t &row, tf::buffer<standing_t> &out) {
          const auto key =
              plane_piece_key<Index>(local_tables.canon_group(row[0])[0]);
          const auto at =
              std::lower_bound(wave_keys.begin(), wave_keys.end(), key);
          if (at != wave_keys.end() && *at == key)
            out.push_back({key, row[0]});
        },
        tf::checked);
  }
  const auto parent_at = [&layout, n_parents](Index group) {
    const auto found =
        std::lower_bound(layout.begin(), layout.end(), group,
                         [](const plane_split_piece_layout<Index> &split,
                            Index id) { return split.parent < id; });
    return found != layout.end() && found->parent == group
               ? std::size_t(found - layout.begin())
               : n_parents;
  };
  // A split must retire its own key. A child equal to that key is an endpoint
  // split the ordering boundary failed to remove.
  if (tf::parallel_contains(
          tf::make_sequence_range(n_parents),
          [&](std::size_t at) {
            const auto &split = layout[at];
            const auto parent_key = plane_piece_key<Index>(
                local_tables.canon_group(split.parent)[0]);
            for (Index rank = 0; rank < split.pieces; ++rank)
              if (plane_piece_key<Index>(proposals[std::size_t(
                      split.first + rank * split.instances)]) == parent_key)
                return true;
            return false;
          },
          tf::checked))
    return false;

  tf::buffer<Index> participants;
  tf::generic_generate(
      tf::make_range(matched), participants,
      [&](const standing_t &row, tf::buffer<Index> &out) {
        if (parent_at(row.group) == n_parents)
          out.push_back(row.group);
      },
      tf::checked);
  tf::generic_generate(
      tf::make_range(changed_groups), participants,
      [&](Index group, tf::buffer<Index> &out) {
        if (parent_at(group) == n_parents)
          out.push_back(group);
      },
      tf::checked);
  if (participants.size() != 0) {
    tbb::parallel_sort(participants.begin(), participants.end());
    participants.erase_till_end(
        std::unique(participants.begin(), participants.end()));
  }
  // Every participant is a local group, and ownership is by the whole carrier
  // closure. Prove that closure before minting or retiring any group.
  if (tf::parallel_contains(
          tf::make_range(participants),
          [&](Index group) {
            for (const auto &def : local_tables.canon_group(group)) {
              const auto plane = plane_of_face(def.face);
              if (plane < Index(0) ||
                  std::size_t(plane) >= plane_ticket.size())
                return true;
              const auto block = plane_ticket[std::size_t(plane)];
              if (block < Index(0) || block >= n_local_blocks)
                return true;
            }
            return false;
          },
          tf::checked))
    return false;
  tf::buffer<Index> participant_prefix;
  participant_prefix.allocate(participants.size() + 1);
  participant_prefix[0] = Index(0);
  {
    std::size_t rows = 0;
    for (std::size_t at = 0; at < participants.size(); ++at) {
      rows += local_tables.canon_group(participants[at]).size();
      if (rows > index_extent - proposals.size())
        return false;
      participant_prefix[at + 1] = Index(rows);
    }
  }

  tf::buffer<def_t> records;
  records.allocate(proposals.size() +
                   std::size_t(participant_prefix[participants.size()]));
  tf::parallel_for_each(
      tf::make_sequence_range(proposals.size()),
      [&](std::size_t slot) {
        auto record = proposals[slot];
        record.id = Index(slot);
        records[slot] = record;
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(participants.size()),
      [&](std::size_t at) {
        auto write = proposals.size() + std::size_t(participant_prefix[at]);
        for (const auto &def : local_tables.canon_group(participants[at])) {
          auto record = def;
          record.id = Index(-1);
          records[write++] = record;
        }
      });
  tbb::parallel_sort(
      records.begin(), records.end(), [](const def_t &x, const def_t &y) {
        if (tf::intersect::graph::plane_def_key_less(x, y))
          return true;
        if (tf::intersect::graph::plane_def_key_less(y, x))
          return false;
        if (tf::intersect::graph::plane_def_instance_less(x, y))
          return true;
        if (tf::intersect::graph::plane_def_instance_less(y, x))
          return false;
        // a child dominates only the standing instance it FULLY restates
        const auto x_standing = x.id == Index(-1);
        const auto y_standing = y.id == Index(-1);
        return std::tie(x_standing, x.id) < std::tie(y_standing, y.id);
      });
  tf::buffer<Index> class_offsets;
  tf::sequenced_generate(
      tf::make_sequence_range(records.size()), class_offsets,
      [&records](std::size_t at, tf::buffer<Index> &out) {
        if (at == 0 ||
            tf::intersect::graph::plane_def_key_less(records[at - 1],
                                                      records[at]))
          out.push_back(Index(at));
      },
      tf::checked);
  class_offsets.push_back(Index(records.size()));
  // groups fuse; instances never do — a true repeat is the WHOLE identity
  const auto same_instance = [](const def_t &x, const def_t &y) {
    return same_plane_piece_definition_instance(x, y) && x.flags == y.flags;
  };
  const auto n_classes = class_offsets.size() - 1;
  const auto class_of = [&records, &class_offsets,
                        n_classes](const key_t &key) {
    std::size_t lo = 0;
    std::size_t hi = n_classes;
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      if (plane_piece_key<Index>(records[std::size_t(class_offsets[mid])]) <
          key)
        lo = mid + 1;
      else
        hi = mid;
    }
    // every matched key and every participant's key IS a class key, so the
    // class is total
    assert(lo < n_classes &&
           plane_piece_key<Index>(records[std::size_t(class_offsets[lo])]) ==
               key);
    return lo;
  };

  tf::buffer<Index> kept_classes;
  if (changed_groups.size() != 0) {
    tf::sequenced_generate(
        tf::make_range(changed_groups), kept_classes,
        [&](Index group, tf::buffer<Index> &out) {
          if (parent_at(group) != n_parents)
            return;
          const auto span = local_tables.canon_group(group);
          const auto cls = class_of(plane_piece_key<Index>(span[0]));
          if (std::size_t(class_offsets[cls + 1] - class_offsets[cls]) ==
              span.size())
            out.push_back(Index(cls));
        },
        tf::checked);
    tbb::parallel_sort(kept_classes.begin(), kept_classes.end());
  }
  const auto group_base = n_canon;
  const auto kept_class = [&kept_classes](std::size_t cls) {
    return std::binary_search(kept_classes.begin(), kept_classes.end(),
                              Index(cls));
  };
  // the mints stay dense over the classes that state one
  const auto mint_of = [&kept_classes, group_base](std::size_t cls) {
    return group_base + Index(cls) -
           Index(std::lower_bound(kept_classes.begin(), kept_classes.end(),
                                  Index(cls)) -
                 kept_classes.begin());
  };

  tf::buffer<Index> counts;
  counts.allocate(n_classes);
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes), [&](std::size_t cls) {
        if (kept_class(cls)) {
          counts[cls] = Index(0);
          return;
        }
        const auto begin = std::size_t(class_offsets[cls]);
        const auto end = std::size_t(class_offsets[cls + 1]);
        Index count = 0;
        for (auto at = begin; at < end; ++at)
          if (at == begin || !same_instance(records[at - 1], records[at]))
            ++count;
        counts[cls] = count;
      });

  tf::buffer<Index> def_prefix;
  def_prefix.allocate(n_classes + 1);
  def_prefix[0] = Index(0);
  for (std::size_t cls = 0; cls < n_classes; ++cls) {
    if (counts[cls] > std::numeric_limits<Index>::max() - def_prefix[cls])
      return false;
    def_prefix[cls + 1] = def_prefix[cls] + counts[cls];
  }
  auto &defs = local_tables.defs();
  if (defs.size() > index_extent - std::size_t(def_prefix[n_classes]) ||
      n_classes > index_extent - std::size_t(n_canon))
    return false;

  final_row.clear();
  losers.clear();
  const auto def_row_base = Index(defs.size());
  const auto n_mints = n_classes - kept_classes.size();
  defs.reallocate(std::size_t(def_row_base) +
                  std::size_t(def_prefix[n_classes]));
  const auto old_group_offsets = def_offsets.size();
  def_offsets.reallocate(old_group_offsets + n_mints);
  final_row.allocate(proposals.size());
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes), [&](std::size_t cls) {
        if (kept_class(cls))
          return;
        const auto begin = std::size_t(class_offsets[cls]);
        const auto end = std::size_t(class_offsets[cls + 1]);
        const auto mint = mint_of(cls);
        auto write = std::size_t(def_row_base + def_prefix[cls]);
        Index row = Index(-1);
        for (auto at = begin; at < end; ++at) {
          if (at == begin || !same_instance(records[at - 1], records[at])) {
            auto def = records[at];
            def.id = mint;
            row = Index(write);
            defs[write++] = def;
          }
          // every slot that stated this instance answers the ONE row it
          // survives as, whether it wrote it or repeated it
          const auto slot = records[at].id;
          if (slot != Index(-1))
            final_row[std::size_t(slot)] = row;
        }
        def_offsets[old_group_offsets + std::size_t(mint - group_base)] =
            Index(write);
      });
  local_tables.n_canon() = group_base + Index(n_mints);

  tf::sequenced_generate(
      tf::make_range(participants), losers,
      [&](Index group, tf::buffer<std::array<Index, 2>> &out) {
        const auto cls = class_of(
            plane_piece_key<Index>(local_tables.canon_group(group)[0]));
        if (!kept_class(cls))
          out.push_back({group, mint_of(cls)});
      },
      tf::checked);
  // A split retires every immutable root that currently routes to its parent.
  // Otherwise the router composes through a fusion loser. Both facts are
  // stated in the router's one wave sweep, with retirement taking precedence.
  if (layout.size() != 0 || losers.size() != 0)
    tf::parallel_for_each(
        tf::make_sequence_range(group_router.size()),
        [&](std::size_t at) {
          const auto route = group_router[at];
          if (route < Index(0))
            return;
          const auto split = std::lower_bound(
              layout.begin(), layout.end(), route,
              [](const plane_split_piece_layout<Index> &piece, Index id) {
                return piece.parent < id;
              });
          if (split != layout.end() && split->parent == route) {
            group_router[at] = Index(-2);
            return;
          }
          const auto found = std::lower_bound(
              losers.begin(), losers.end(), route,
              [](const std::array<Index, 2> &loser, Index id) {
                return loser[0] < id;
              });
          if (found != losers.end() && (*found)[0] == route)
            group_router[at] = (*found)[1];
        },
        tf::checked);
  return true;
}

} // namespace tf::arrangement
