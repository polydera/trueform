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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>

namespace tf::arrangement {

template <typename Index>
using plane_generation_source_ticket = std::array<Index, 3>;

template <typename Index>
constexpr Index plane_generation_immutable_source = Index(0);

template <typename Index>
constexpr Index plane_generation_current_source = Index(1);

/// Build the dense global-plane -> local-generation ticket.
template <typename Index, typename GlobalPlanes>
auto make_plane_generation_ticket(const GlobalPlanes &global_planes,
                                  Index global_plane_extent,
                                  tf::buffer<Index> &ticket) -> bool {
  if (tf::parallel_contains(
          tf::make_sequence_range(global_planes.size()),
          [&](std::size_t local) {
            const auto global = global_planes[local];
            return global < Index(0) || global >= global_plane_extent ||
                   (local != 0 && global <= global_planes[local - 1]);
          },
          tf::checked))
    return false;
  ticket.allocate(std::size_t(global_plane_extent));
  tf::parallel_fill(ticket, Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(Index(global_planes.size())),
      [&](Index local) {
        ticket[std::size_t(global_planes[std::size_t(local)])] = local;
      },
      tf::checked);
  return true;
}

/// Copy the effective active plane set into one clean canonical source.
///
/// A dense global-plane ticket chooses exactly one authority per carrier:
/// current PA state when present, otherwise immutable LA. Records carry their
/// source tier and group by value, are rewritten through the closed identity
/// table, and sort by final endpoint key. Equal-key runs therefore retain the
/// union of all definition instances and every immutable ancestor. The source
/// dense ticket tables map each immediate source group to its clean output
/// group. They are barrier-local proposal translators, not a second persistent
/// read path.
template <typename Index, typename Int, typename Immutable,
          typename GlobalPlanes, typename VertexOffsets>
auto make_plane_generation_source(
    const Immutable &immutable,
    const tf::intersect::graph::plane_tables<Index, Int> &current,
    const tf::buffer<Index> &current_plane_ticket,
    const tf::buffer<Index> &current_ancestor_offsets,
    const tf::buffer<Index> &current_ancestors,
    const GlobalPlanes &global_planes, Index global_plane_extent,
    const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets,
    tf::intersect::graph::plane_tables<Index, Int> &output,
    tf::buffer<Index> &global_plane_of_local,
    tf::buffer<Index> &ancestor_offsets, tf::buffer<Index> &ancestors,
    tf::buffer<Index> &immutable_group_ticket,
    tf::buffer<Index> &current_group_ticket) -> bool {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  struct record_t {
    std::array<Index, 4> key;
    char flipped;
    Index source_kind;
    Index source_group;
    Index source_row;
    /// the record's seat in plane order, which the key sort leaves behind
    Index home;
  };

  if (current_ancestor_offsets.size() == 0) {
    if (current.n_canon() != Index(0))
      return false;
  } else {
    if (current_ancestor_offsets.size() != std::size_t(current.n_canon()) + 1 ||
        current_ancestor_offsets[0] != Index(0) ||
        current_ancestor_offsets[current_ancestor_offsets.size() - 1] !=
            Index(current_ancestors.size()))
      return false;
    if (tf::parallel_contains(
            tf::make_sequence_range(current_ancestor_offsets.size() - 1),
            [&](std::size_t group) {
              return current_ancestor_offsets[group] < Index(0) ||
                     current_ancestor_offsets[group] >
                         current_ancestor_offsets[group + 1];
            },
            tf::checked))
      return false;
  }

  const auto n_planes = Index(global_planes.size());
  if (tf::parallel_contains(
          tf::make_sequence_range(global_planes.size()),
          [&](std::size_t local) {
            const auto global = global_planes[local];
            if (global < Index(0) || global >= global_plane_extent ||
                (local != 0 && global <= global_planes[local - 1]))
              return true;
            if (std::size_t(global) < current_plane_ticket.size() &&
                current_plane_ticket[std::size_t(global)] != Index(-1)) {
              const auto source = current_plane_ticket[std::size_t(global)];
              return source < Index(0) ||
                     std::size_t(source) >= current.edges().size();
            }
            return global >= immutable.n_planes();
          },
          tf::checked))
    return false;

  tf::buffer<Index> next_global_planes;
  next_global_planes.allocate(std::size_t(n_planes));
  tf::parallel_copy(global_planes, next_global_planes);

  const auto state_plane_records = [&](Index local, const auto &state) {
    const auto global = next_global_planes[std::size_t(local)];
    const bool from_current =
        std::size_t(global) < current_plane_ticket.size() &&
        current_plane_ticket[std::size_t(global)] != Index(-1);
    const auto emit = [&](const auto &block, bool local_block) {
      for (const auto row : block) {
        const auto immutable_row = !local_block || row < Index(0);
        const auto source_kind = immutable_row
                                     ? plane_generation_immutable_source<Index>
                                     : plane_generation_current_source<Index>;
        const auto source_row =
            immutable_row ? (local_block ? -1 - row : row) : row;
        const auto &parent =
            immutable_row
                ? immutable.edge_defs()[std::size_t(source_row)]
                : current.edge_defs()[std::size_t(source_row)];
        const auto ends = tf::intersect::graph::plane_merged_endpoints(
            merges, vertex_offsets, parent);
        if (ends.lo_tag == ends.hi_tag && ends.lo_id == ends.hi_id)
          continue;
        std::array<Index, 2> a{Index(ends.lo_tag), ends.lo_id};
        std::array<Index, 2> b{Index(ends.hi_tag), ends.hi_id};
        const bool flipped = b < a;
        if (flipped)
          std::swap(a, b);
        state(std::array<Index, 4>{a[0], a[1], b[0], b[1]}, flipped,
              source_kind, parent.id, source_row);
      }
    };
    if (from_current) {
      const auto source = current_plane_ticket[std::size_t(global)];
      // a block of this arrangement's tier states rows in the FLAT row space:
      // AN UNCHANGED GROUP STAYS THE WORLD'S, so the row itself says which
      // table holds the parent
      emit(current.plane_edges(source), true);
    } else {
      emit(immutable.plane_edges(global), false);
    }
  };

  // THE PLANE IS THE BLOCK: its records are stated once, and the offsets the
  // aggregation builds are the plane CSR itself. A carrier space of none still
  // states its one offset.
  auto &plane_offsets = output.edges().offsets_buffer();
  plane_offsets.allocate(1);
  plane_offsets[0] = Index(0);
  tf::buffer<record_t> records;
  tf::generate_offset_blocks(
      tf::make_sequence_range(n_planes), plane_offsets, records,
      [&](Index local, tf::buffer<record_t> &out) {
        state_plane_records(local, [&out](const std::array<Index, 4> &key,
                                          bool flipped, Index source_kind,
                                          Index source_group,
                                          Index source_row) {
          out.push_back({key, char(flipped), source_kind, source_group,
                         source_row, Index(0)});
        });
      },
      tf::checked);
  // the seat is the flat position the offsets just fixed
  tf::parallel_for_each(
      tf::make_sequence_range(records.size()),
      [&records](std::size_t at) { records[at].home = Index(at); },
      tf::checked);
  tbb::parallel_sort(records.begin(), records.end(),
                     [](const record_t &x, const record_t &y) {
                       return std::tie(x.key, x.source_kind, x.source_group,
                                       x.source_row) <
                              std::tie(y.key, y.source_kind, y.source_group,
                                       y.source_row);
                     });

  auto &defs = output.defs();
  auto &def_offsets = output.def_offsets();
  auto &plane_data = output.edges().data_buffer();
  defs.allocate(records.size());
  def_offsets.clear();
  if (records.size() == 0) {
    def_offsets.push_back(Index(0));
    output.n_canon() = Index(0);
  } else {
    tf::compute_offsets(
        records, std::back_inserter(def_offsets), Index(0),
        [](const record_t &x, const record_t &y) { return x.key == y.key; });
    output.n_canon() = Index(def_offsets.size()) - Index(1);
  }

  tf::buffer<plane_generation_source_ticket<Index>> all_tickets;
  all_tickets.allocate(records.size());
  tf::parallel_for_each(
      tf::make_sequence_range(output.n_canon()),
      [&](Index group) {
        const auto begin = std::size_t(def_offsets[std::size_t(group)]);
        const auto end = std::size_t(def_offsets[std::size_t(group) + 1]);
        for (auto row = begin; row < end; ++row) {
          const auto parent = [&]() -> def_t {
            if (records[row].source_kind ==
                plane_generation_current_source<Index>)
              return current.edge_defs()[std::size_t(records[row].source_row)];
            return immutable.edge_defs()[std::size_t(records[row].source_row)];
          }();
          auto def = parent;
          const std::array<Index, 2> lo{records[row].key[0],
                                        records[row].key[1]};
          const std::array<Index, 2> hi{records[row].key[2],
                                        records[row].key[3]};
          tf::intersect::graph::plane_piece_def(
              def, records[row].flipped ? hi : lo,
              records[row].flipped ? lo : hi, parent, Index(1));
          def.id = group;
          defs[row] = def;
          all_tickets[row] = {records[row].source_kind,
                              records[row].source_group, group};
        }
      },
      tf::checked);

  plane_data.allocate(records.size());
  tf::parallel_for_each(
      tf::make_sequence_range(records.size()),
      [&](std::size_t row) {
        plane_data[std::size_t(records[row].home)] = Index(row);
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index plane) {
        std::sort(plane_data.begin() +
                      std::size_t(plane_offsets[std::size_t(plane)]),
                  plane_data.begin() +
                      std::size_t(plane_offsets[std::size_t(plane) + 1]));
      },
      tf::checked);

  tbb::parallel_sort(all_tickets.begin(), all_tickets.end());
  all_tickets.erase_till_end(
      std::unique(all_tickets.begin(), all_tickets.end()));
  tf::buffer<Index> next_immutable_group_ticket;
  tf::buffer<Index> next_current_group_ticket;
  next_immutable_group_ticket.allocate(std::size_t(immutable.n_canon()));
  next_current_group_ticket.allocate(std::size_t(current.n_canon()));
  tf::parallel_fill(next_immutable_group_ticket, Index(-1));
  tf::parallel_fill(next_current_group_ticket, Index(-1));
  // the tickets are sorted and uniqued, so one source group stated with two
  // clean groups is an adjacent pair
  if (tf::parallel_contains(
          tf::make_sequence_range(all_tickets.size()),
          [&](std::size_t row) {
            const auto &ticket = all_tickets[row];
            const auto &dense =
                ticket[0] == plane_generation_immutable_source<Index>
                    ? next_immutable_group_ticket
                    : next_current_group_ticket;
            if (ticket[1] < Index(0) || std::size_t(ticket[1]) >= dense.size())
              return true;
            return row != 0 && all_tickets[row - 1][0] == ticket[0] &&
                   all_tickets[row - 1][1] == ticket[1];
          },
          tf::checked))
    return false;
  tf::parallel_for_each(
      all_tickets,
      [&](const plane_generation_source_ticket<Index> &ticket) {
        auto &dense = ticket[0] == plane_generation_immutable_source<Index>
                          ? next_immutable_group_ticket
                          : next_current_group_ticket;
        dense[std::size_t(ticket[1])] = ticket[2];
      },
      tf::checked);

  using ancestry_record_t = std::array<Index, 2>;
  tf::buffer<ancestry_record_t> ancestry_records;
  tf::generic_generate(
      all_tickets, ancestry_records,
      [&](const auto &ticket, tf::buffer<ancestry_record_t> &out) {
        if (ticket[0] == plane_generation_immutable_source<Index>) {
          out.push_back({ticket[2], ticket[1]});
          return;
        }
        const auto source = std::size_t(ticket[1]);
        for (auto ancestor = current_ancestor_offsets[source];
             ancestor < current_ancestor_offsets[source + 1]; ++ancestor)
          out.push_back({ticket[2], current_ancestors[std::size_t(ancestor)]});
      },
      tf::checked);
  tbb::parallel_sort(ancestry_records.begin(), ancestry_records.end());
  ancestry_records.erase_till_end(
      std::unique(ancestry_records.begin(), ancestry_records.end()));
  ancestor_offsets.allocate_and_initialize(std::size_t(output.n_canon()) + 1,
                                           Index(0));
  for (const auto &record : ancestry_records)
    ++ancestor_offsets[std::size_t(record[0]) + 1];
  for (std::size_t group = 1; group < ancestor_offsets.size(); ++group)
    ancestor_offsets[group] += ancestor_offsets[group - 1];
  ancestors.allocate(ancestry_records.size());
  tf::parallel_for_each(
      tf::make_sequence_range(ancestry_records.size()),
      [&](std::size_t row) { ancestors[row] = ancestry_records[row][1]; },
      tf::checked);
  global_plane_of_local = std::move(next_global_planes);
  immutable_group_ticket = std::move(next_immutable_group_ticket);
  current_group_ticket = std::move(next_current_group_ticket);
  return true;
}

} // namespace tf::arrangement
