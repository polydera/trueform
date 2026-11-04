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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/tetrahedra_buffer.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/build_parallel_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/recovery/recover_tet_face_phase.hpp"
#include "../../topology/cdt/dt3/recovery/recover_tet_segment_phase.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_recovery.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_status.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_phase.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_status.hpp"
#include "../../topology/cdt/dt3/tet_claim_workspace.hpp"
#include "../../topology/cdt/dt3/tet_execution_tuning.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/cdt/dt3/tet_facet_key.hpp"
#include "../../topology/cdt/dt3/tet_facets.hpp"
#include "../../topology/cdt/dt3/tet_minted_site.hpp"
#include "../../topology/cdt/dt3/tet_position.hpp"
#include "../../topology/cdt/dt3/tet_shared_owners.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/cdt/dt3/tet_state.hpp"
#include "../../topology/cdt/dt3/tetrahedralization_owner.hpp"
#include "../carve/carve_tet_complex.hpp"
#include "../entrance/count_tet_open_edges.hpp"
#include "../entrance/tet_entrance.hpp"
#include "../entrance/tet_entrance_budget.hpp"
#include "../tet_mesh_report.hpp"
#include "../tet_mesh_status.hpp"
#include "./count_flat_tet_cells.hpp"
#include "./count_inverted_tets.hpp"
#include "./count_negative_tet_cells.hpp"
#include "./count_tet_deep_sites.hpp"
#include "./emit_tet_face_children.hpp"
#include "./emit_tet_mesh.hpp"
#include "./state_tet_face_facets.hpp"
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace tf::tetmesh {

/// Mesh the solid `polygons` winds from its `entrance`: the Delaunay build
/// over the sites, phase S over the edges bounding the entrance's facets
/// and then phase F over the facets, carving by winding, and emission of the
/// positively wound cells in `RealOut`, with the triangles covering each facet,
/// the facets each input face lies in and the report. A constraint the frame
/// splits too coarsely deepens past it: a segment deepens its own frame and
/// a face places its point below the frame, as far as the deep lattice
/// admits. The surface's own corners and the rung's widths are the caller's
/// to state in `report`.
template <typename RealOut, typename Policy, typename Index, typename RealT,
          typename Int, typename Coord>
auto mesh_tet_entrance(const tf::polygons<Policy> &polygons,
                       const tet_entrance<Index, RealT, Int, Coord> &entrance,
                       tf::tet_mesh_report report) {
  namespace dt3 = tf::topology::cdt::dt3;
  using face_t = std::array<Index, 3>;
  using owner_t = dt3::tetrahedralization_owner<Index, Coord, Int>;
  tf::tetrahedra_buffer<Index, RealOut> mesh;
  tf::offset_block_buffer<Index, face_t> children;
  tf::offset_block_buffer<Index, Index> face_facets;
  report.frame_bits = entrance.frame_bits;
  report.entrance_passes = entrance.surface.passes;
  report.dropped_degenerate = entrance.surface.dropped_degenerate;
  report.stitched_triangles = entrance.surface.stitched_triangles;
  report.input_bits = entrance.surface.input_bits;
  report.storage_bits = tet_entrance_budget<Int, Coord>::storage_bits;
  report.entrance_failed_planes = entrance.surface.failed_planes;
  report.entrance_failed_faces = entrance.surface.failed_faces;
  report.entrance_open_edges = entrance.surface.open_edges.unbalanced;
  if (entrance.surface.failed_faces) {
    report.status = tf::tet_mesh_status::refused;
    return std::make_tuple(std::move(mesh), std::move(children),
                           std::move(face_facets), report);
  }
  if (entrance.surface.open_edges.unbalanced) {
    report.status = entrance.surface.open_edges.boundary
                        ? tf::tet_mesh_status::open
                        : tf::tet_mesh_status::unwound;
    report.input_open_edges = count_tet_open_edges(polygons.faces()).unbalanced;
    return std::make_tuple(std::move(mesh), std::move(children),
                           std::move(face_facets), report);
  }
  owner_t owner;
  dt3::tet_claim_workspace<Index> work{};
  const auto &points = entrance.surface.points;
  const int frame_bits = entrance.frame_bits;
  const auto sites = tf::make_mapped_range(
      tf::make_sequence_range(points.size()),
      [&points, frame_bits](std::size_t i) {
        return dt3::tet_minted_site<Index, Coord>{
            Index(i), dt3::tet_minted_position<Coord>(points[i], frame_bits)};
      });
  if (!dt3::build_parallel_tetrahedralization(owner, sites, work)) {
    report.status =
        owner._refusal == tf::tetrahedralization_refusal::index_capacity
            ? tf::tet_mesh_status::capacity
            : tf::tet_mesh_status::rank_deficient;
    return std::make_tuple(std::move(mesh), std::move(children),
                           std::move(face_facets), report);
  }
  std::vector<face_t> faces(entrance.surface.triangles.size());
  const auto &slot = owner._index_map.f();
  tf::parallel_for_each(
      tf::make_sequence_range(faces.size()),
      [&faces, &entrance, &slot](std::size_t t) {
        for (std::size_t k = 0; k < 3; ++k)
          faces[t][k] = slot[std::size_t(entrance.surface.triangles[t][k])];
      },
      tf::checked);
  const auto facets =
      tf::make_offset_block_range(entrance.surface.facets, faces);
  const std::size_t n_facets = facets.size();
  dt3::tet_segment_phase<Index, Coord> phase;
  dt3::tet_face_recovery<Index, Coord, Int> recovery;
  phase.segment.site_budget =
      dt3::tet_site_count(owner) * dt3::tet_execution_tuning::site_budget_ratio;
  phase.segment.deepens = true;
  const auto refuse = [&report, &phase] {
    report.refused_segments = report.refused_depth = report.refused_budget = 0;
    for (auto status : phase.segment.status) {
      report.refused_segments += status != dt3::tet_segment_status::recovered;
      report.refused_depth += status == dt3::tet_segment_status::depth;
      report.refused_budget += status == dt3::tet_segment_status::budget;
    }
  };
  const auto capacity = [&] {
    report.status = tf::tet_mesh_status::capacity;
    return std::make_tuple(std::move(mesh), std::move(children),
                           std::move(face_facets), report);
  };
  assert(dt3::tet_site_count(owner) == entrance.surface.points.size());
  if (!dt3::recover_tet_segment_phase<true>(
          owner, work, std::vector<std::array<Index, 2>>{}, facets,
          entrance.frame_bits, phase, entrance.surface.link))
    return capacity();
  if (!dt3::recover_tet_face_phase<true>(owner, work, facets, phase, recovery))
    return capacity();
  refuse();
  report.census = {std::size_t(phase.segment.round), phase.segment.shell_splits,
                   report.refused_segments, phase.segment.unrestored,
                   recovery.census};
  report.census.flip_recovered = phase.segment.flip_recovered;
  report.census.search_recovered = phase.segment.search_recovered;
  report.census.split_rows = phase.segment.split_rows;
  report.census.hull_barred = phase.segment.flips.hull_barred;
  for (auto status : recovery.status)
    report.refused_faces += status != dt3::tet_face_status::recovered;
  report.refused_depth +=
      recovery.census.standing + recovery.census.off_lattice;
  const auto deep = count_tet_deep_sites(owner);
  report.deep_sites = deep[0];
  report.deepest = int(deep[1]);
  tet_carving<Index> carving;
  carve_tet_complex(
      owner,
      tf::make_mapped_range(tf::make_sequence_range(n_facets),
                            [&facets](std::size_t f) { return facets[f][0]; }),
      carving);
  const bool recovered = !report.refused_segments && !report.refused_faces;
  report.status = recovered
                      ? (carving.consistent ? tf::tet_mesh_status::meshed
                                            : tf::tet_mesh_status::unwound)
                      : tf::tet_mesh_status::refused;
  if (!carving.consistent) {
    report.input_open_edges = count_tet_open_edges(polygons.faces()).unbalanced;
    return std::make_tuple(std::move(mesh), std::move(children),
                           std::move(face_facets), report);
  }

  tf::buffer<char> used;
  used.allocate(dt3::tet_site_count(owner));
  tf::parallel_fill(used, char(0));
  const auto recovered_face = [&recovery](Index f) {
    return recovery.status[std::size_t(f)] == dt3::tet_face_status::recovered;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(std::min(std::size_t(owner._corners.size()),
                                       owner._protected_slots.size())),
      [&owner, &used, &recovered_face](std::size_t cell) {
        if (owner._states[cell] == dt3::tet_state::dead ||
            !owner._protected_slots[cell])
          return;
        for (std::size_t slot = 0; slot < 4; ++slot) {
          if (!dt3::tet_facet_is_protected(owner, Index(cell), slot))
            continue;
          const Index who = owner._owners[4 * cell + slot];
          if (who == owner_t::none)
            continue;
          bool recovered = false;
          if (who == owner_t::shared_owner) {
            const auto stack = dt3::tet_shared_owners(
                owner, dt3::tet_facet_key(owner._corners[cell], slot));
            for (auto at = stack.first; at != stack.second; ++at)
              recovered = recovered || recovered_face(at->second);
          } else {
            recovered = recovered_face(who);
          }
          if (recovered)
            for (auto site : dt3::tet_facet(owner._corners[cell], slot))
              used[std::size_t(site)] = 1;
        }
      },
      tf::checked);
  tf::buffer<Index> point_of;
  mesh = emit_tet_mesh<RealOut, Index>(
      owner, [&carving](std::size_t cell) { return carving.winding[cell] > 0; },
      used, entrance.converter, entrance.frame_bits, point_of);
  report.inverted = count_inverted_tets(mesh);
  report.flat = count_flat_tet_cells(owner, [&carving](std::size_t cell) {
    return carving.winding[cell] > 0;
  });
  report.negative_cells = count_negative_tet_cells(carving);

  emit_tet_face_children<Index>(owner, facets, recovery.status, point_of,
                                children);
  state_tet_face_facets(polygons.faces().size(), entrance.surface.facets,
                        entrance.surface.parents, face_facets);
  return std::make_tuple(std::move(mesh), std::move(children),
                         std::move(face_facets), report);
}

} // namespace tf::tetmesh
