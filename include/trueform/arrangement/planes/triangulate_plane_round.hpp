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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/deal_jobs.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/cdt/constrained_delaunay_full_span_alias.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../../topology/constrained_delaunay_triangulator.hpp"
#include "../../topology/topo_id.hpp"
#include "./emit_plane_fan.hpp"
#include "./emit_plane_triangles.hpp"
#include "./find_plane_carrier_fan.hpp"
#include "./plane_arrangement_arena.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_member_statements.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_statement.hpp"
#include "./plane_round_evidence.hpp"
#include "./plane_triangulation_types.hpp"
#include "./prepare_plane_triangulation.hpp"
#include "./resolve_plane_refusal.hpp"
#include "./run_plane_cdt.hpp"
#include "./run_refined_plane_cdt.hpp"
#include "tbb/parallel_sort.h"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// WAVE. Triangulate the round's planes. The plane is the independent grain
/// and its position is its identity, so the aggregator's walk in input order
/// is the whole join. A later round carries the frontier the wave published;
/// the stock round carries every plane there is.
template <typename Index, typename Int, typename World, typename Planes,
          typename VertexOffsets, typename PointOfFlat>
auto triangulate_plane_round(
    const World &world, const Planes &planes,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, const VertexOffsets &vertex_offsets,
    Index n_flat, Index crossing_kind, bool refined, bool record_arrangement,
    bool record_cells, const tf::cdt_refine_config &refine_config,
    const tf::offset_block_buffer<Index, tf::exact::pt3<Int>> &steiner_sites,
    const PointOfFlat &point_of_flat,
    plane_arrangement_arena<Index, Int> &output,
    plane_round_evidence<Index, Int> &evidence) -> void {
  using pt3_t = tf::exact::pt3<Int>;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using coplanar_t = tf::arrangement::coplanar_descriptor<Index>;
  struct work_t {
    tf::constrained_delaunay_triangulator<Index, Int, Int> cdt;
    tf::cdt_refiner<Index, Int, Int> refiner;
    tf::buffer<Index> ends;
    tf::buffer<Index> corner_local;
    tf::points_buffer<Int, 2> pts2;
    tf::buffer<Index> cons;
    tf::buffer<char> bnd;
    tf::buffer<Index> cons_group;
    tf::buffer<std::array<Index, 2>> cons_row;
    tf::buffer<plane_member_statement<Index>> cons_statements;
    tf::buffer<plane_member_statement<Index>> member_statements;
    tf::buffer<plane_member_statement<Index>> folded_members;
    tf::buffer<Index> statement_offsets;
    tf::buffer<short> cons_side;
    tf::buffer<std::array<Index, 2>> member_ticket;
    tf::buffer<tf::topo_id<short>> point_subs;
    tf::buffer<Index> subs_touched;
    tf::buffer<tf::topology::cdt::constrained_delaunay_full_span_alias<Index>>
        cons_aliases;
    tf::buffer<std::array<Index, 2>> cons_alias_blocks;
    tf::buffer<char> cons_boundary_promotions;
    tf::buffer<Index> cons_root;
    tf::buffer<Index> rep;
    tf::buffer<Index> steiner_ticket;
    tf::buffer<pt3_t> steiner_points;
    tf::buffer<std::array<Index, 3>> tris;
    tf::buffer<std::array<Index, 3>> slots;
    tf::buffer<std::array<tf::topo_id<short>, 3>> subs;
    tf::buffer<Index> coplanar_of;
    tf::buffer<char> stacked;
    tf::buffer<coplanar_t> coplanar;
    tf::buffer<Index> cell_of;
    tf::buffer<Index> face_cell;
    tf::buffer<std::array<Index, 3>> cell_walk;
    tf::buffer<Index> cell_stack;
    tf::buffer<std::array<Index, 2>> face_range;
    tf::buffer<Index> planes;
    tf::buffer<std::array<Index, 2>> range;
    tf::buffer<std::array<Index, 2>> cop_range;
    tf::buffer<std::array<Index, 2>> stn_range;
    tf::buffer<Index> refused;
    tf::buffer<plane_flat_weld<Index>> welds;
    tf::buffer<plane_recovery_statement<Index, param_t>> statements;
    tf::buffer<plane_recovery_proposal<Index, param_t>> topology;
    tf::buffer<std::array<Index, 3>> region_adj;
    tf::buffer<Index> region_offsets;
    tf::buffer<Index> region_stack;
    tf::buffer<Index> surv_mi;
    tf::buffer<Index> surv_pos;
    tf::buffer<char> region_reached;
    tf::buffer<std::array<Index, 2>> cover_range;
    tf::buffer<Index> cover;
    tf::buffer<Index> member_offsets;
    tf::buffer<Index> member_cursor;
    tf::buffer<Index> member_tri;
    tf::buffer<char> member_rev;
    int face_orientation = 1;
    plane_arrangement_census census;
  };
  struct block_start_t {
    std::size_t coplanar;
    Index descriptor;
    Index triangle;
  };
  tf::buffer<block_start_t> block_starts;
  // A refusal states its crossings against the canonical group that roots
  // them, so a world with no group space cannot be asked: the round records
  // the refusal, the barrier makes the tier real, and the carrier is read
  // again against it.
  const bool states_groups = world.materialized();
  auto task = [&](auto &&range, work_t &local) {
    for (const auto plane : range) {
      const auto before = local.tris.size();
      // the arrangement facts are a request, so their stores roll back to
      // their own marks and a build that asked for none rolls back nothing
      const auto arrangement_before = local.slots.size();
      const auto cop_before = local.coplanar.size();
      const auto face_before = local.face_range.size();
      const auto weld_before = local.welds.size();
      const auto steiner_before = local.steiner_points.size();
      const auto n_members = std::size_t(world.member_count(plane));
      const bool pooled = n_members > 1;
      bool ok = prepare_plane_triangulation(world, local_tables, plane_ticket,
                                            vertex_offsets, plane, local,
                                            point_of_flat);
      bool plane_refined = false;
      bool fanned = false;
      if (ok) {
        local.census.constraints += local.bnd.size();
        if (pooled)
          ++local.census.stacks;
        // THE CONVEX FAMILY: a carrier whose constraint set is one simple
        // ring it can prove convex answers with the fan, and no triangulation
        // is built. A stack is answered per member and a refined carrier is
        // owed interior points, so neither is one of the family.
        if (!pooled && !refined) {
          const auto fan = find_plane_carrier_fan<Index, Int>(local);
          fanned = fan.size != 0;
          if (fanned) {
            emit_plane_fan<Index, Int>(local, fan, record_arrangement,
                                       record_cells);
            ++local.census.fanned_planes;
          }
        }
        if (!fanned) {
          if (refined) {
            plane_refined = run_refined_plane_cdt(world, steiner_sites, plane,
                                                  local, pooled, refine_config);
            if (!plane_refined)
              ++local.census.refined_fallbacks;
          }
          ok = plane_refined || run_plane_cdt(local, pooled, false);
        }
      }
      if (ok && !fanned)
        ok = plane_refined
                 ? emit_plane_triangles<Index, Int>(
                       world, plane, n_flat, local, local.refiner, pooled,
                       record_arrangement, record_cells)
                 : emit_plane_triangles<Index, Int>(
                       world, plane, n_flat, local, local.cdt, pooled,
                       record_arrangement, record_cells);
      if (!ok) {
        local.tris.erase_till_end(local.tris.begin() + std::ptrdiff_t(before));
        local.slots.erase_till_end(local.slots.begin() +
                                   std::ptrdiff_t(arrangement_before));
        local.subs.erase_till_end(local.subs.begin() + std::ptrdiff_t(before));
        local.coplanar_of.erase_till_end(local.coplanar_of.begin() +
                                         std::ptrdiff_t(arrangement_before));
        local.stacked.erase_till_end(local.stacked.begin() +
                                     std::ptrdiff_t(arrangement_before));
        local.coplanar.erase_till_end(local.coplanar.begin() +
                                      std::ptrdiff_t(cop_before));
        local.welds.erase_till_end(local.welds.begin() +
                                   std::ptrdiff_t(weld_before));
        local.steiner_points.erase_till_end(local.steiner_points.begin() +
                                            std::ptrdiff_t(steiner_before));
        if (record_cells)
          local.cell_of.erase_till_end(local.cell_of.begin() +
                                       std::ptrdiff_t(before));
        local.refused.push_back(plane);
        local.face_range.erase_till_end(local.face_range.begin() +
                                        std::ptrdiff_t(face_before));
        for (std::size_t m = 0; m < n_members; ++m)
          local.face_range.push_back({Index(before), Index(before)});
        if (states_groups)
          resolve_plane_refusal(world, local_tables, plane_ticket,
                                vertex_offsets, n_flat, crossing_kind, plane,
                                local, point_of_flat);
      }
      local.planes.push_back(plane);
      local.range.push_back({Index(before), Index(local.tris.size())});
      local.cop_range.push_back(
          {Index(cop_before), Index(local.coplanar.size())});
      local.stn_range.push_back(
          {Index(steiner_before), Index(local.steiner_points.size())});
    }
  };
  auto aggregate = [&](const work_t &local, const tf::none_t &) {
    const auto base = Index(output.tris.size());
    const auto cop_base = Index(output.coplanar.size());
    const auto stn_base = Index(output.steiners.size());
    if (record_arrangement)
      block_starts.push_back({output.coplanar_of.size(), cop_base, base});
    tf::core::append(local.steiner_points, output.steiners);
    tf::core::append(local.tris, output.tris);
    tf::core::append(local.slots, output.slots);
    tf::core::append(local.subs, output.subs);
    tf::core::append(local.coplanar_of, output.coplanar_of);
    tf::core::append(local.stacked, output.stacked);
    tf::core::append(local.coplanar, output.coplanar);
    tf::core::append(local.cell_of, output.cell_of);
    tf::core::append(local.welds, evidence.welds);
    tf::core::append(local.statements, evidence.statements);
    tf::core::append(local.topology, evidence.topology);
    std::size_t face_position = 0;
    for (std::size_t i = 0; i < local.planes.size(); ++i) {
      const auto plane = local.planes[i];
      output.range[std::size_t(plane)] = {local.range[i][0] + base,
                                          local.range[i][1] + base};
      output.cop_range[std::size_t(plane)] = {local.cop_range[i][0] + cop_base,
                                              local.cop_range[i][1] + cop_base};
      output.stn_range[std::size_t(plane)] = {local.stn_range[i][0] + stn_base,
                                              local.stn_range[i][1] + stn_base};
      const auto n_members = std::size_t(world.member_count(plane));
      for (std::size_t m = 0; m < n_members; ++m) {
        const auto range = local.face_range[face_position++];
        output.face_range[std::size_t(world.member(plane, Index(m)))] = {
            range[0] + base, range[1] + base};
      }
    }
    for (const auto plane : local.refused)
      evidence.refused.push_back(plane);
    evidence.census += local.census;
  };
  // The deal's weight reads the definition tables, so a world that has not
  // materialized them keeps the plain partition — its carriers are one face
  // each and uniform, which is the one shape contiguous chunking serves.
  if (refined || states_groups) {
    tf::buffer<Index> jobs;
    tf::deal_jobs<Index>(
        planes,
        [&](Index plane) {
          return plane_definition_edge_count(world, local_tables, plane_ticket,
                                             plane);
        },
        jobs);
    tf::blocked_reduce_sequenced_aggregate(tf::make_range(jobs), tf::none,
                                           work_t{}, task, aggregate);
    // the deal permutes the jobs; the refusal set stays the ascending
    // unique set its consumers close rounds and publish failures with
    if (evidence.refused.size() != 0)
      tbb::parallel_sort(evidence.refused.begin(), evidence.refused.end());
  } else {
    tf::blocked_reduce_sequenced_aggregate(planes, tf::none, work_t{}, task,
                                           aggregate);
  }
  if (!record_arrangement)
    return;
  block_starts.push_back({output.coplanar_of.size(),
                          Index(output.coplanar.size()),
                          Index(output.tris.size())});
  // Both tickets are task-local. Aggregation records their ordered append
  // bases; independent task blocks translate once after the barrier.
  tf::parallel_for_each(tf::make_sequence_range(block_starts.size() - 1),
                        [&](std::size_t block) {
                          const auto &start = block_starts[block];
                          const auto &next = block_starts[block + 1];
                          for (auto t = start.coplanar; t < next.coplanar; ++t)
                            if (output.coplanar_of[t] != Index(-1))
                              output.coplanar_of[t] += start.descriptor;
                          for (auto d = std::size_t(start.descriptor);
                               d < std::size_t(next.descriptor); ++d)
                            output.coplanar[d].survivor += start.triangle;
                        });
}

} // namespace tf::arrangement
