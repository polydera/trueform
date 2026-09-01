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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_piece_definition_instance.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"
#include "./propose_plane_split_pieces.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>

namespace tf::arrangement {

/// Carry the canonicalize's identities into the blocks that name them.
///
/// A carrier of a split parent is REBUILT: the parent's rows replaced by that
/// carrier's child rows — the canonicalize's final rows, read through the
/// proposal slots — the index list ordered again by KEY, appended as a NEW
/// block, the ticket repointed. The list orders by key because that is the
/// invariant every block consumer indexes by, and mint order is not key order
/// once a created-first child joins a tier the world built key-major.
///
/// A BLOCK NAMES EACH DEFINITION INSTANCE ONCE. That is the mirror of the
/// canonicalize's own law — groups fuse, instances never do — read from the
/// side that names them: two of a carrier's slots may read ONE final row,
/// because two parents it carries state the same piece the moment they
/// coincide, and a block is the instances the carrier constrains, not the
/// slots that proposed them. So the rebuilt list is the SET of the rows its
/// old rows state; the reserved extent is the upper bound one row per slot,
/// and the block's true extent is the fuse's own, stated once here. Nothing
/// else can break the invariant: the port carries a world row to its local
/// rank one for one, the weld drops rows and rewrites definitions where they
/// lie, and the substitution below moves an instance's row from the loser's
/// name to the winner's — a row this block would already name only if it
/// named that instance twice, which this law forbids.
///
/// A carrier of a structurally retired LOSER that the rebuild does not touch
/// is SUBSTITUTED IN PLACE: the loser's row becomes the winner's row of the
/// same provenance. The winner carries the loser's key, so the block's order
/// is preserved by construction — no re-sort, no re-triangulation, no new
/// block. The slot's provisional identity in that carrier's standing product
/// stays as it was, and stays harmless: an active carrier resolves by KEY,
/// and the key did not move. That is also why this substitution owes a
/// REFUSED carrier no retry, unlike a weld: the key it reads is the same key,
/// so the constraint set it never triangulated is unchanged.
///
/// False rejects the call and publishes nothing: the local tier's interior
/// was proven by the producers that appended it, so this states only what the
/// rebuild itself indexes. The ownership law requires every loser's carrier
/// to hold a local block before any block is changed.
template <typename Index, typename Int, typename PlaneOfFace>
auto commit_plane_wave_blocks(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::buffer<plane_split_piece_layout<Index>> &layout,
    const tf::buffer<Index> &final_row, const tf::buffer<Index> &carrier_planes,
    const tf::buffer<std::array<Index, 2>> &losers,
    const PlaneOfFace &plane_of_face,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket) -> bool {
  const auto n_parents = layout.size();
  const auto n_losers = losers.size();
  const auto n_canon = local_tables.n_canon();
  const auto n_blocks = local_tables.edges().size();
  if (!local_tables.well_formed())
    return false;
  const auto tier =
      make_plane_tier_definitions(world_tables, local_tables, true);
  const auto &def_offsets = local_tables.def_offsets();
  // A row still the world's states a group this wave never took, so no split
  // and no retirement can reach it: it survives the rebuild as it is.
  const auto local_group_of = [&tier](Index row) {
    return tier.immutable(row) ? Index(-1) : tier[std::size_t(row)].id;
  };
  {
    std::size_t slots = 0;
    for (std::size_t at = 0; at < n_parents; ++at) {
      const auto &split = layout[at];
      if (split.parent < Index(0) || split.parent >= n_canon ||
          split.pieces < Index(2) || split.instances < Index(1) ||
          (at != 0 && layout[at - 1].parent >= split.parent) ||
          split.first != Index(slots))
        return false;
      slots += std::size_t(split.pieces) * std::size_t(split.instances);
    }
    if (final_row.size() != slots)
      return false;
  }
  if (tf::parallel_contains(
          tf::make_range(final_row),
          [&](Index row) { return !tier.contains(row) || tier.immutable(row); },
          tf::checked))
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(n_losers),
          [&](std::size_t at) {
            const auto &loser = losers[at];
            return loser[0] < Index(0) || loser[0] >= n_canon ||
                   loser[1] < Index(0) || loser[1] >= n_canon ||
                   loser[0] == loser[1] ||
                   def_offsets[std::size_t(loser[1]) + 1] <=
                       def_offsets[std::size_t(loser[1])] ||
                   (at != 0 && losers[at - 1][0] >= loser[0]);
          },
          tf::checked))
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(carrier_planes.size()),
          [&](std::size_t at) {
            const auto plane = carrier_planes[at];
            if (plane < Index(0) || std::size_t(plane) >= plane_ticket.size() ||
                (at != 0 && carrier_planes[at - 1] >= plane))
              return true;
            const auto block = plane_ticket[std::size_t(plane)];
            return block < Index(0) || std::size_t(block) >= n_blocks;
          },
          tf::checked))
    return false;
  if (n_parents == 0 && n_losers == 0)
    return true;

  // every rewritten row asks which split owns its group, twice — the count
  // pass and the rewrite — and group ids are dense, so the layout answers
  // through a ticket
  tf::buffer<Index> parent_ticket;
  parent_ticket.allocate(std::size_t(n_canon));
  tf::parallel_fill(parent_ticket, Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(n_parents),
      [&](std::size_t at) {
        parent_ticket[std::size_t(layout[at].parent)] = Index(at);
      },
      tf::checked);
  const auto loser_at = [&losers, n_losers](Index group) {
    const auto found =
        std::lower_bound(losers.begin(), losers.end(), group,
                         [](const std::array<Index, 2> &loser, Index id) {
                           return loser[0] < id;
                         });
    return found != losers.end() && (*found)[0] == group
               ? std::size_t(found - losers.begin())
               : n_losers;
  };
  // the winner's span is ascending by the WHOLE oriented instance —
  // provenance then flags, because A->B and B->A on one key are two facts —
  // so one search states which of its rows supersedes a retired one
  const auto winner_row =
      [&tier, &def_offsets](
          Index winner,
          const tf::intersect::graph::plane_edge_def<Index> &retired) {
        auto lo = std::size_t(def_offsets[std::size_t(winner)]);
        auto hi = std::size_t(def_offsets[std::size_t(winner) + 1]);
        while (lo < hi) {
          const auto mid = lo + (hi - lo) / 2;
          if (tf::intersect::graph::plane_def_instance_less(tier[mid],
                                                            retired))
            lo = mid + 1;
          else
            hi = mid;
        }
        assert(lo < std::size_t(def_offsets[std::size_t(winner) + 1]) &&
               same_plane_piece_definition_instance(tier[lo], retired) &&
               tier[lo].flags == retired.flags);
        return Index(lo);
      };

  // a loser's carriers, gathered before the rebuild moves any ticket
  tf::buffer<std::array<Index, 3>> substitutions;
  if (n_losers != 0) {
    tf::generic_generate(tf::make_range(losers), substitutions,
                         [&](const std::array<Index, 2> &loser,
                             tf::buffer<std::array<Index, 3>> &out) {
                           for (const auto &def :
                                local_tables.canon_group(loser[0]))
                             out.push_back({plane_of_face(def.face), loser[0],
                                            loser[1]});
                         },
                         tf::checked);
    tbb::parallel_sort(substitutions.begin(), substitutions.end());
    substitutions.erase_till_end(
        std::unique(substitutions.begin(), substitutions.end()));
    if (tf::parallel_contains(
            tf::make_range(substitutions),
            [&](const std::array<Index, 3> &substitution) {
              if (substitution[0] < Index(0) ||
                  std::size_t(substitution[0]) >= plane_ticket.size())
                return true;
              const auto block =
                  plane_ticket[std::size_t(substitution[0])];
              return block < Index(0) || std::size_t(block) >= n_blocks;
            },
            tf::checked))
      return false;
  }

  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  auto &block_offsets = local_tables.edges().offsets_buffer();
  auto &block_rows = local_tables.edges().data_buffer();
  const auto block_base = Index(n_blocks);
  const auto row_base = Index(block_rows.size());
  tf::buffer<std::size_t> row_prefix;
  row_prefix.allocate(carrier_planes.size() + 1);
  row_prefix[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(carrier_planes.size()), [&](std::size_t at) {
        std::size_t count = 0;
        for (const auto row : local_tables.plane_edges(
                 plane_ticket[std::size_t(carrier_planes[at])])) {
          const auto group = local_group_of(row);
          const auto parent =
              group == Index(-1) ? Index(-1)
                                 : parent_ticket[std::size_t(group)];
          const auto added =
              parent == Index(-1)
                  ? std::size_t(1)
                  : std::size_t(layout[std::size_t(parent)].pieces);
          count = count > index_extent - added ? index_extent : count + added;
        }
        row_prefix[at + 1] = count;
      });
  for (std::size_t at = 0; at < carrier_planes.size(); ++at) {
    if (row_prefix[at + 1] > index_extent - row_prefix[at])
      return false;
    row_prefix[at + 1] += row_prefix[at];
  }
  if (block_rows.size() >
          index_extent - row_prefix[carrier_planes.size()] ||
      n_blocks > index_extent - carrier_planes.size())
    return false;
  const auto old_block_offsets = block_offsets.size();
  block_offsets.reallocate(old_block_offsets + carrier_planes.size());
  block_rows.reallocate(std::size_t(row_base) +
                        row_prefix[carrier_planes.size()]);
  tf::buffer<std::size_t> kept_prefix;
  kept_prefix.allocate(carrier_planes.size() + 1);
  kept_prefix[0] = 0;
  struct block_local_t {
    tf::buffer<std::array<Index, 5>> keyed;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(carrier_planes.size()),
      [&](std::size_t at, block_local_t &local) {
        const auto plane = carrier_planes[at];
        const auto begin = std::size_t(row_base) + row_prefix[at];
        auto write = begin;
        for (const auto row :
             local_tables.plane_edges(plane_ticket[std::size_t(plane)])) {
          const auto &def = tier[std::size_t(row)];
          const auto group = local_group_of(row);
          const auto parent =
              group == Index(-1) ? Index(-1)
                                 : parent_ticket[std::size_t(group)];
          if (parent != Index(-1)) {
            const auto &split = layout[std::size_t(parent)];
            const auto position = row - def_offsets[std::size_t(split.parent)];
            for (Index rank = 0; rank < split.pieces; ++rank)
              block_rows[write++] = final_row[std::size_t(
                  split.first + rank * split.instances + position)];
            continue;
          }
          const auto loser = group == Index(-1) ? n_losers : loser_at(group);
          block_rows[write++] =
              loser == n_losers ? row : winner_row(losers[loser][1], def);
        }
        sort_plane_block_by_key(tier, block_rows, begin, write, local.keyed);
        // the order is key then row, so every slot that read one fused row
        // stands with its twin and the set is one adjacent sweep
        auto kept = begin;
        for (auto read = begin; read != write; ++read)
          if (kept == begin || block_rows[kept - 1] != block_rows[read])
            block_rows[kept++] = block_rows[read];
        kept_prefix[at + 1] = kept - begin;
        plane_ticket[std::size_t(plane)] = block_base + Index(at);
      },
      block_local_t{});
  // The reserved extent was one row per slot; the blocks close up over the
  // rows their sets actually name. The kept rows move to a destination OF
  // THEIR OWN: a block whose predecessors dropped rows starts before its own
  // source, so closing up in place would let it write over rows the block
  // before it has not read yet.
  for (std::size_t at = 0; at < carrier_planes.size(); ++at)
    kept_prefix[at + 1] += kept_prefix[at];
  tf::buffer<Index> kept_rows;
  kept_rows.allocate(kept_prefix[carrier_planes.size()]);
  tf::parallel_for_each(
      tf::make_sequence_range(carrier_planes.size()), [&](std::size_t at) {
        const auto source = std::size_t(row_base) + row_prefix[at];
        const auto target = kept_prefix[at];
        const auto rows = kept_prefix[at + 1] - target;
        for (std::size_t row = 0; row < rows; ++row)
          kept_rows[target + row] = block_rows[source + row];
        block_offsets[old_block_offsets + at] =
            Index(std::size_t(row_base) + target + rows);
      });
  block_rows.erase_till_end(block_rows.begin() + std::ptrdiff_t(row_base));
  tf::core::append(kept_rows, block_rows);

  // the independent carrier is the PLANE: one owner rewrites one block,
  // resolving every loser it carries — two losers on one plane must never
  // become two callbacks over the same rows
  tf::buffer<Index> substitution_offsets;
  if (substitutions.size() != 0)
    tf::compute_offsets(
        substitutions, std::back_inserter(substitution_offsets), Index(0),
        [](const std::array<Index, 3> &x, const std::array<Index, 3> &y) {
          return x[0] == y[0];
        });
  tf::parallel_for_each(
      tf::make_sequence_range(substitution_offsets.size() == 0
                                  ? std::size_t(0)
                                  : substitution_offsets.size() - 1),
      [&](std::size_t run) {
        const auto begin = std::size_t(substitution_offsets[run]);
        const auto end = std::size_t(substitution_offsets[run + 1]);
        const auto plane = substitutions[begin][0];
        if (std::binary_search(carrier_planes.begin(), carrier_planes.end(),
                               plane))
          return;
        const auto block = plane_ticket[std::size_t(plane)];
        for (auto at = std::size_t(block_offsets[std::size_t(block)]);
             at < std::size_t(block_offsets[std::size_t(block) + 1]); ++at) {
          const auto group = local_group_of(block_rows[at]);
          if (group == Index(-1))
            continue;
          const auto &def = tier[std::size_t(block_rows[at])];
          auto lo = begin;
          auto hi = end;
          while (lo < hi) {
            const auto mid = lo + (hi - lo) / 2;
            if (substitutions[mid][1] < group)
              lo = mid + 1;
            else
              hi = mid;
          }
          if (lo != end && substitutions[lo][1] == group)
            block_rows[at] = winner_row(substitutions[lo][2], def);
        }
      });
  return true;
}

} // namespace tf::arrangement
