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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/return_report.hpp"
#include "../../core/tetrahedra_buffer.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/take.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../reindex/return_index_map.hpp"
#include "../../topology/cdt/dt3/build_parallel_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/check_tetrahedralization_validity.hpp"
#include "../../topology/cdt/dt3/compact_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/publish_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/recovery/recover_tet_face_phase.hpp"
#include "../../topology/cdt/dt3/recovery/recover_tet_segment_phase.hpp"
#include "../../topology/cdt/dt3/recovery/retire_tet_segments.hpp"
#include "../../topology/cdt/dt3/recovery/revive_tet_segments.hpp"
#include "../../topology/cdt/dt3/recovery/tet_crossing_due.hpp"
#include "../../topology/cdt/dt3/recovery/tet_crossing_pair.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_recovery.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_rim_sites.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_status.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_phase.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_status.hpp"
#include "../../topology/cdt/dt3/recovery/tet_stamps.hpp"
#include "../../topology/cdt/dt3/recovery/walk_to_tet_point.hpp"
#include "../../topology/cdt/dt3/resolve_tet_retired_sites.hpp"
#include "../../topology/cdt/dt3/retire_tet_faces.hpp"
#include "../../topology/cdt/dt3/tet_claim_workspace.hpp"
#include "../../topology/cdt/dt3/tet_facet_owner.hpp"
#include "../../topology/cdt/dt3/tet_facets.hpp"
#include "../../topology/cdt/dt3/tet_minted_site.hpp"
#include "../../topology/cdt/dt3/tet_owner_complex.hpp"
#include "../../topology/cdt/dt3/tet_position.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/cdt/dt3/tet_split_route.hpp"
#include "../../topology/cdt/dt3/tetrahedralization_owner.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/make_face_membership.hpp"
#include "../../topology/make_manifold_edge_link.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../cut/collapse_tet_steiner_corners.hpp"
#include "../cut/cut_tet_pairs.hpp"
#include "../cut/elect_tet_created_site.hpp"
#include "../cut/tet_pair_workspace.hpp"
#include "../entrance/count_tet_open_edges.hpp"
#include "../entrance/merge_tet_facets.hpp"
#include "../entrance/state_tet_entrance_surface.hpp"
#include "../entrance/tet_entrance.hpp"
#include "../entrance/tet_entrance_budget.hpp"
#include "../entrance/tet_entrance_workspace.hpp"
#include "../entrance/tet_facet_workspace.hpp"
#include "../identity/account_tet_identities.hpp"
#include "../identity/tet_identity_authority.hpp"
#include "../regions/wind_tet_regions.hpp"
#include "../tet_mesh_config.hpp"
#include "../tet_mesh_index_map.hpp"
#include "../tet_mesh_report.hpp"
#include "../tet_mesh_status.hpp"
#include "./compact_tet_facets.hpp"
#include "./count_flat_tet_cells.hpp"
#include "./count_inverted_tets.hpp"
#include "./count_negative_tet_cells.hpp"
#include "./count_tet_deep_sites.hpp"
#include "./emit_tet_face_children.hpp"
#include "./emit_tet_mesh.hpp"
#include "./gather_tet_face_owners.hpp"
#include "./measure_tet_recovered_openness.hpp"
#include "./restore_tet_kept_cells.hpp"
#include "./run_tet_census_rounds.hpp"
#include "./state_tet_face_facets.hpp"
#include "./tet_drain_measure.hpp"
#include "./tet_face_map_workspace.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::tetmesh {

/// Mesh the solid `polygons` winds from its `entrance`: the Delaunay build
/// over the sites, phase S over the edges bounding the entrance's facets
/// and then phase F over the facets, region labeling by winding, the
/// restoration of the kept cells to Delaunay (@ref restore_tet_kept_cells), and
/// emission of the positively wound cells in `RealOut`, with the status, or
/// under a `ReportTag` of `tf::return_report_t` the report and the
/// diagnostics only it measures, and under a `ReturnTag` of
/// `tf::return_index_map_t` the @ref tf::tet_mesh_index_map, which
/// `tf::none_t` neither builds nor returns. A
/// constraint the frame splits too coarsely deepens past it: a segment deepens
/// its own frame and a face places its point below the frame, as far as the
/// deep lattice admits. A crossing the recovery's walk meets is named to the
/// phases' cut where it refuses, and the crossings a pass named are cut in a
/// drain on the standing complex (@ref cut_tet_pairs): the curves through
/// them traced and their triangles split, whose created points a piece
/// holds elect the sites standing at them or are minted
/// (@ref elect_tet_created_site), a point no piece holds being no site; the
/// drain re-enters phase S on the pieces' rims and the segments refused for
/// depth, and phase F on the pieces and the faces still refused, continuing
/// both on the rows as they stand: the first wave compacts after each phase,
/// a drain compacts nothing, and the complex is compacted once after the last
/// drain. The complex is canonical where its order is a product, the build's
/// publish and the emit, and keeps its order between. A created point in
/// the cell of the reach grid a site an earlier drain minted for the same
/// pair of input faces stands in is that site, fused into it, its rows
/// naming the point the site stands for, so every drain that mints mints
/// in a cell no earlier mint of its crossing stands in. THE DRAINS' MEASURE
/// (@ref tet_drain_measure): a drain either mints, consuming the one site
/// budget, or strictly reduces the crossings it was handed, so the drains
/// are finite — at most the budget's minting drains, and between two of
/// them no more drains than the crossings handed to the first. They end
/// where no crossing is named, or a drain retires no facet; a drain that
/// neither mints nor reduces is refused as stalled, mints the budget cannot
/// hold refuse for the budget, and a wave that fails or a mint the complex
/// withdraws refuses the run, each counted in the census.
/// Coincident sites weld in the build, to the lowest name, and a segment splits
/// at a site standing on it, so a seam the entrance's triangles leave open may
/// close in the recovery: where the entrance counts an open edge and every
/// constraint recovered, the openness is read on the recovered surface
/// (@ref measure_tet_recovered_openness), a hole refused as `open` and any
/// other free edge as `unwound`. The sites the complex may hold are
/// `config.site_budget_ratio` per entrance site, one budget the recovery's
/// splits and face points and the drains' mints all consume.
/// corners and the rung's widths are the caller's to state in `report`.
template <typename RealOut, typename Policy, typename Index, typename RealT,
          typename Int, typename Coord, typename ReportTag, typename ReturnTag>
auto mesh_tet_entrance(const tf::polygons<Policy> &polygons,
                       tet_entrance<Index, RealT, Int, Coord> entrance,
                       tf::tet_mesh_report report,
                       const tf::tet_mesh_config &config, ReportTag,
                       ReturnTag) {
  constexpr bool reports = std::is_same_v<ReportTag, tf::return_report_t>;
  namespace dt3 = tf::topology::cdt::dt3;
  using face_t = std::array<Index, 3>;
  using owner_t = dt3::tetrahedralization_owner<Index, Coord, Int>;
  tf::tetrahedra_buffer<Index, RealOut> mesh;
  tf::tet_mesh_index_map<Index> map;
  dt3::tet_segment_phase<Index, Coord> phase;
  dt3::tet_face_recovery<Index, Coord, Int> recovery;
  // what stands at the end, read off the statuses
  const auto refuse = [&report, &phase, &recovery] {
    report.refused_segments = report.refused_depth = report.refused_budget = 0;
    report.refused_faces = 0;
    for (auto status : phase.segment.status) {
      report.refused_segments += status != dt3::tet_segment_status::recovered &&
                                 status != dt3::tet_segment_status::retired;
      report.refused_depth += status == dt3::tet_segment_status::depth;
      report.refused_budget += status == dt3::tet_segment_status::budget;
    }
    for (std::size_t f = 0; f < recovery.status.size(); ++f) {
      const bool refused =
          recovery.status[f] != dt3::tet_face_status::recovered &&
          recovery.status[f] != dt3::tet_face_status::retired;
      report.refused_faces += refused;
      report.refused_depth += refused && recovery.deep[f];
    }
  };
  // every count read once off its producer: the surface's accumulators,
  // the statuses, and the recovery's cumulative census
  const auto finish = [&] {
    refuse();
    report.dropped_degenerate = entrance.surface.dropped_degenerate;
    report.stitched_triangles = entrance.surface.stitched_triangles;
    report.census.segment_rounds = std::size_t(phase.segment.round);
    report.census.shell_splits = phase.segment.shell_splits;
    report.census.segments_refused = report.refused_segments;
    report.census.segments_capped = phase.segment.passes_capped;
    report.census.region_tasks = phase.segment.region_tasks;
    report.census.region_deferred = phase.segment.region_deferred;
    report.census.unrestored_facets = phase.segment.unrestored;
    report.census.faces = recovery.census;
    report.census.flip_recovered = phase.segment.flip_recovered;
    report.census.search_recovered = phase.segment.search_recovered;
    report.census.split_rows = phase.segment.split_rows;
    report.census.hull_barred = phase.segment.flips.hull_barred;
    report.census.restored_before_faces = phase.segment.restored_before_faces;
    if constexpr (std::is_same_v<ReportTag, tf::return_report_t>) {
      if constexpr (std::is_same_v<ReturnTag, tf::return_index_map_t>)
        return std::make_tuple(std::move(mesh), report, std::move(map));
      else
        return std::make_tuple(std::move(mesh), report);
    } else if constexpr (std::is_same_v<ReturnTag, tf::return_index_map_t>)
      return std::make_tuple(std::move(mesh), report.status, std::move(map));
    else
      return std::make_tuple(std::move(mesh), report.status);
  };
  report.frame_bits = entrance.frame_bits;
  report.input_bits = entrance.surface.input_bits;
  report.storage_bits = tet_entrance_budget<Int, Coord>::storage_bits;
  tf::buffer<tet_edge_step<Index>> steps;
  auto &surface = entrance.surface;
  const auto &points = surface.points;
  if (points.size() == 0)
    return finish();
  owner_t owner;
  dt3::tet_claim_workspace<Index> work{};
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
    return finish();
  }
  dt3::publish_tetrahedralization<true>(owner);
  tf::core::std_vector<face_t> faces(surface.triangles.size());
  const auto &slot = owner._index_map.f();
  const std::size_t built = points.size();
  // the one producer of the drain's identities: the entrance's points first,
  // each seated at the site the build stood it at
  tet_identity_authority<Index, Int> authority;
  authority.name_originals(points, slot);
  const auto slot_of = [&authority](Index p) { return authority.slot_of(p); };
  // every identity is live, retired or vanished against the complex as it
  // stands: no published cell names the site of a retired identity
  const auto account = [&] {
    if constexpr (tf::core::verification)
      assert_tet_identities_are_accounted(
          authority,
          tf::take(owner._corners.data_buffer(), 4 * owner._n_finite),
          Index(dt3::tet_site_count(owner)));
  };
  tf::parallel_for_each(
      tf::make_sequence_range(faces.size()),
      [&faces, &surface, &slot](std::size_t t) {
        for (std::size_t k = 0; k < 3; ++k)
          faces[t][k] = slot[std::size_t(surface.triangles[t][k])];
      },
      tf::checked);
  const auto facets_of = [&surface, &faces] {
    return tf::make_offset_block_range(surface.facets, faces);
  };
  phase.segment.site_budget =
      dt3::tet_site_count(owner) * config.site_budget_ratio;
  tet_drain_measure measure;
  measure.budget = phase.segment.site_budget;
  phase.segment.deepens = true;
  phase.segment.reach =
      Coord(surface.converter.quantum()) * (Coord(1) << frame_bits);
  const auto capacity = [&] {
    report.status = tf::tet_mesh_status::capacity;
    return finish();
  };
  if constexpr (tf::core::verification)
    for (std::size_t i = 0; i < built; ++i)
      tf::core::verify(
          slot[i] != owner_t::none &&
              dt3::tet_site_position(owner, slot[i]) ==
                  dt3::tet_minted_position<Coord>(points[i], frame_bits),
          "every entrance point is the site standing at its position, "
          "coincident ones welded into one");
  tf::buffer<dt3::tet_crossing_pair<Index>> named;
  const auto name_crossing = [&named](const dt3::tet_crossing_pair<Index> &pair,
                                      dt3::tet_crossing_due<Index> &) {
    named.push_back(pair);
    return false;
  };
  dt3::tet_stamps near_stamps;
  tf::buffer<Index> near_cells, census_due, census_moved, convicted;
  tet_corner_census_scratch<Index, Coord> census_work;
  const auto compact = [&] {
    dt3::compact_tetrahedralization<true, false>(owner);
    if constexpr (tf::core::verification)
      tf::core::verify(dt3::check_tetrahedralization_validity(
                           owner,
                           [&census_work](Index slot) {
                             return tet_census_standing(census_work, slot) !=
                                    slot;
                           }),
                       "the recovery leaves a valid tetrahedralization");
  };
  // the census after a face phase: a face refused with nothing to cut
  // convicts the recovery's sites at its corners, which collapse into them,
  // and the faces moved are recovered again, to a round that collapses
  // nothing; the walk names afresh what the faces it reads still cross
  const auto census = [&]() -> bool {
    census_moved.clear();
    const auto round = [&]() -> tet_census_round {
      census_due.clear();
      convicted.clear();
      const auto made = collapse_tet_steiner_corners(
          owner, work, facets_of(), named, phase.segment, census_work,
          census_due, convicted);
      report.census.collapsed += made[0];
      report.census.collapse_refused += made[1];
      if (!made[0])
        return {};
      dt3::resolve_tet_retired_sites(owner, [&census_work](Index slot) {
        return tet_census_standing(census_work, slot);
      });
      for (auto f : convicted)
        census_due.push_back(f);
      std::sort(census_due.begin(), census_due.end());
      census_due.erase_till_end(
          std::unique(census_due.begin(), census_due.end()));
      std::size_t kept = 0;
      for (std::size_t i = 0; i < named.size(); ++i)
        if (named[i].face < 0 ||
            !std::binary_search(census_due.begin(), census_due.end(),
                                named[i].face))
          named[kept++] = named[i];
      named.erase_till_end(named.begin() + std::ptrdiff_t(kept));
      for (auto f : census_due)
        census_moved.push_back(f);
      if (!dt3::recover_tet_face_phase<true>(owner, work, facets_of(),
                                             census_due, phase, recovery,
                                             name_crossing))
        return {made[0], census_due.size(), false};
      compact();
      return {made[0], census_due.size(), true};
    };
    return run_tet_census_rounds(dt3::tet_execution_tuning::census_rounds,
                                 round, report.census.collapse_refused);
  };
  if (!dt3::recover_tet_segment_phase<true>(
          owner, work, tf::core::std_vector<std::array<Index, 2>>{},
          facets_of(), tf::make_sequence_range(Index(facets_of().size())),
          frame_bits, phase, surface.link, name_crossing))
    return capacity();
  compact();
  if (!dt3::recover_tet_face_phase<true>(
          owner, work, facets_of(),
          tf::make_sequence_range(Index(facets_of().size())), phase, recovery,
          name_crossing))
    return capacity();
  compact();
  if (!census())
    return capacity();
  const tf::exact::pt_converter<Int, Int, 3> identity{
      {surface.converter.quantum()}};
  const double band = double(surface.band);
  tf::buffer<Index> retired, candidates, held;
  tf::buffer<std::array<Index, 2>> beside;
  tf::buffer<signed char> dead;
  tf::buffer<dt3::tet_retired_facet<Index>> retiring;
  tf::blocked_buffer<Index, 3> rows;
  tf::buffer<Index> sources;
  tf::manifold_edge_link<Index, 3> due_link;
  tet_pair_workspace<Index, Int> cut_work;
  tet_entrance_workspace<Index, Int> state_work;
  tf::buffer<dt3::tet_minted_site<Index, Coord>> minted;
  tf::buffer<Index> minted_ids;
  tf::buffer<Index> due_faces, refused_faces, depth_segments, walked, requeued;
  for (std::size_t f = 0; f < recovery.status.size(); ++f)
    if (recovery.status[f] == dt3::tet_face_status::refused)
      refused_faces.push_back(Index(f));
  for (std::size_t g = 0; g < phase.segment.status.size(); ++g)
    if (phase.segment.status[g] == dt3::tet_segment_status::depth)
      depth_segments.push_back(Index(g));
  for (;;) {
    std::size_t crossings = 0;
    for (const auto &pair : named)
      crossings += pair.segment >= 0 ? std::size_t(pair.crosses.segment >= 0) +
                                           std::size_t(pair.crosses.face >= 0)
                                     : std::size_t(1);
    report.pairs_unresolved = crossings;
    const auto opened = measure.open(crossings);
    if (opened == tet_drain_end::converged)
      break;
    if (opened == tet_drain_end::stalled) {
      ++report.census.stalled_drains;
      report.status = tf::tet_mesh_status::refused;
      return finish();
    }
    report.pairs_named += crossings;
    if (config.split_constraints) {
      dead.reallocate_and_initialize(surface.triangles.size(),
                                     static_cast<signed char>(0));
      const auto live = [&dead](std::size_t t) { return !dead[t]; };
      const std::size_t standing_facets = facets_of().size();
      const auto drained = cut_tet_pairs<Index, Int>(
          surface.points, surface.triangles, faces, dt3::tet_site_count(owner),
          surface.facets, surface.parents, live, phase.segments, named,
          identity, band, retired, rows, sources, beside, authority, cut_work);
      named.clear();
      report.failed_triangles = drained.failed;
      const auto cut = measure.cut(drained.failed, retired.size());
      if (cut == tet_drain_end::failed) {
        report.status = tf::tet_mesh_status::refused;
        return finish();
      }
      if (cut == tet_drain_end::unretired)
        break;
      // every point a piece holds that no site stands for elects its site
      minted.clear();
      minted_ids.clear();
      {
        dt3::tet_owner_complex<true, owner_t, dt3::tet_claim_workspace<Index>>
            complex(owner, work);
        for (const auto &[id, from] : beside) {
          // a point the census convicted into a corner speaks the corner
          if (authority.target(id) != Index(-1))
            continue;
          const auto position = dt3::tet_minted_position<Coord>(
              authority.position_of(id), frame_bits);
          const Index hint = owner._site_cells.load(std::size_t(slot_of(from)));
          const auto seat = elect_tet_created_site(
              complex, position,
              dt3::walk_to_tet_point(complex, position, hint), authority,
              authority.pair_of(id), phase.segment.reach, near_stamps,
              near_cells);
          if (seat.fused)
            authority.retire(id, authority.identity_at(seat.site));
          else if (seat.site != owner_t::none)
            authority.seat(id, seat.site);
          else {
            minted.push_back(
                {Index(complex.next_name() + minted.size()), position});
            minted_ids.push_back(id);
          }
        }
      }
      authority.close();
      // a point fused into an earlier mint is that mint: its rows speak it
      for (auto &corner : rows.data_buffer())
        corner = authority.standing(corner);
      // the mints consume the one site budget
      if (measure.mint(dt3::tet_site_count(owner), minted.size()) ==
          tet_drain_end::budget) {
        ++report.census.budget_drains;
        report.status = tf::tet_mesh_status::refused;
        return finish();
      }
      ++report.drains;
      report.pairs_cut += crossings;
      report.triangles_cut += drained.cut;
      report.census.retired_crossing += drained.retired_crossing;
      report.census.retired_neighbour += drained.retired_neighbour;
      report.pairs_unresolved = 0;
      for (auto f : retired) {
        recovery.status[std::size_t(f)] = dt3::tet_face_status::retired;
        for (Index t = surface.facets[std::size_t(f)];
             t < surface.facets[std::size_t(f) + 1]; ++t)
          dead[std::size_t(t)] = 1;
      }
      dt3::retire_tet_faces(owner, retired, dt3::tet_face_rim_sites(phase),
                            retiring);
      dt3::retire_tet_segments(phase, recovery.status, retired, candidates,
                               held);
      {
        dt3::tet_owner_complex<true, owner_t, dt3::tet_claim_workspace<Index>>
            complex(owner, work);
        const std::size_t base = slot.size();
        complex.record_killed_edges(true);
        if (minted.size() &&
            !complex.append_sites(tf::make_range(minted), dt3::tet_split_route))
          return capacity();
        report.census.revived_segments += dt3::revive_tet_segments(
            complex, phase.segment, phase.segment.revisits);
        complex.record_killed_edges(false);
        // a mint the complex withdrew holds no site, and the faces that name
        // it cannot stand: the drain is refused
        std::size_t withdrawn = 0;
        for (std::size_t i = 0; i < minted_ids.size(); ++i) {
          if (slot[base + i] == owner_t::none)
            ++withdrawn;
          else
            authority.seat(minted_ids[i], slot[base + i]);
        }
        if (measure.seat(withdrawn) == tet_drain_end::withdrawn) {
          recovery.census.unplaced += withdrawn;
          report.status = tf::tet_mesh_status::refused;
          return finish();
        }
        if constexpr (tf::core::verification)
          for (const auto id : minted_ids)
            tf::core::verify(authority.slot_of(id) != Index(-1),
                             "a drain leaves no point at slot -1");
      }
      account();
      {
        const auto cut = tf::make_polygons(rows, surface.points.points());
        const auto cut_membership = tf::make_face_membership<Index>(cut);
        const auto cut_link =
            tf::make_manifold_edge_link<Index>(cut | tf::tag(cut_membership));
        const auto peers = [&cut_link](std::size_t t) { return cut_link[t]; };
        std::array<std::size_t, 3> thrown{};
        Index n_new = 0;
        {
          tet_facet_workspace<Index, Int> merge;
          n_new = merge_tet_facets<Int, Index>(
              surface.points.points(), rows, cut_membership, peers, merge,
              state_work.triangles, state_work.origin, state_work.facet_of,
              thrown);
        }
        state_tet_entrance_surface(surface, n_new, thrown, sources, peers,
                                   state_work, due_link);
        if constexpr (tf::core::verification) {
          tf::buffer<Index> live_ids;
          const std::size_t n_triangles = surface.triangles.size();
          for (std::size_t t = 0; t < n_triangles; ++t)
            if (t >= dead.size() || !dead[t])
              live_ids.push_back(Index(t));
          const auto stated = count_tet_open_edges(
              tf::make_indirect_range(live_ids, surface.triangles), steps);
          tf::core::verify((stated.unbalanced != 0) ==
                               (surface.open_edges.unbalanced != 0),
                           "a drain neither opens nor closes the surface");
        }
      }
      const std::size_t standing_triangles = faces.size();
      faces.resize(surface.triangles.size());
      for (std::size_t t = standing_triangles; t < faces.size(); ++t)
        for (std::size_t k = 0; k < 3; ++k)
          faces[t][k] = Index(slot_of(surface.triangles[t][k]));
      if constexpr (tf::core::verification)
        for (std::size_t t = standing_triangles; t < faces.size(); ++t)
          for (std::size_t k = 0; k < 3; ++k)
            tf::core::verify(faces[t][k] != owner_t::none,
                             "every point a drain names is a site");
      due_faces.clear();
      for (auto f : refused_faces)
        if (recovery.status[std::size_t(f)] == dt3::tet_face_status::refused)
          due_faces.push_back(f);
      for (std::size_t f = standing_facets; f < facets_of().size(); ++f)
        due_faces.push_back(Index(f));
      auto &segment = phase.segment;
      walked.clear();
      for (auto k : segment.revisits)
        walked.push_back(segment.subsegments[std::size_t(k)].segment);
      requeued.clear();
      for (auto g : depth_segments)
        if (segment.status[std::size_t(g)] == dt3::tet_segment_status::depth) {
          segment.status[std::size_t(g)] = dt3::tet_segment_status::pending;
          walked.push_back(g);
          [[maybe_unused]] std::size_t steps = 0;
          for (Index k = segment.first[std::size_t(g)]; k != Index(-1);
               k = segment.subsegments[std::size_t(k)].next) {
            requeued.push_back(k);
            if constexpr (tf::core::verification)
              tf::core::verify(++steps <= segment.subsegments.size(),
                               "a segment's chain is acyclic");
          }
        }
      std::sort(requeued.begin(), requeued.end());
      for (auto k : requeued)
        segment.revisits.push_back(k);
      const std::size_t standing_segments = segment.status.size();
      if (!dt3::recover_tet_segment_phase<true>(
              owner, work, tf::core::std_vector<std::array<Index, 2>>{},
              facets_of(),
              tf::make_sequence_range(Index(standing_facets),
                                      Index(facets_of().size())),
              frame_bits, phase, due_link, name_crossing))
        return capacity();
      if (!dt3::recover_tet_face_phase<true>(owner, work, facets_of(),
                                             due_faces, phase, recovery,
                                             name_crossing))
        return capacity();
      if (!census())
        return capacity();
      for (auto f : census_moved)
        due_faces.push_back(f);
      std::sort(due_faces.begin(), due_faces.end());
      due_faces.erase_till_end(std::unique(due_faces.begin(), due_faces.end()));
      for (std::size_t g = standing_segments; g < segment.status.size(); ++g)
        walked.push_back(Index(g));
      std::sort(walked.begin(), walked.end());
      walked.erase_till_end(std::unique(walked.begin(), walked.end()));
      depth_segments.clear();
      for (auto g : walked)
        if (segment.status[std::size_t(g)] == dt3::tet_segment_status::depth)
          depth_segments.push_back(g);
      refused_faces.clear();
      for (auto f : due_faces)
        if (recovery.status[std::size_t(f)] == dt3::tet_face_status::refused)
          refused_faces.push_back(f);
    } else
      break;
  }
  if (report.drains)
    compact();
  const auto facets = facets_of();
  const std::size_t n_facets = facets.size();
  refuse();
  if constexpr (reports) {
    const auto deep = count_tet_deep_sites(owner);
    report.deep_sites = deep[0];
    report.deepest = int(deep[1]);
  }
  const bool recovered = !report.refused_segments && !report.refused_faces;
  tf::buffer<dt3::tet_facet_owner<Index>> face_owners;
  if (recovered && entrance.surface.open_edges.unbalanced) {
    tet_face_map_workspace<Index> surface_work;
    tf::face_membership<Index> membership;
    tf::offset_block_buffer<Index, face_t> surface;
    const auto openness = measure_tet_recovered_openness<Index>(
        owner, facets, recovery.status, face_owners, surface_work, steps,
        membership, surface);
    report.recovered_open_edges = openness.edges.unbalanced;
    if (openness.edges.unbalanced) {
      report.status = openness.holed ? tf::tet_mesh_status::open
                                     : tf::tet_mesh_status::unwound;
      if constexpr (reports)
        report.input_open_edges =
            count_tet_open_edges(polygons.faces(), steps).unbalanced;
      return finish();
    }
  }
  const auto covering =
      tf::make_mapped_range(tf::make_sequence_range(n_facets),
                            [&facets](std::size_t f) { return facets[f][0]; });
  tet_regions<Index> regions;
  wind_tet_regions(owner, covering, regions);
  report.status = recovered
                      ? (regions.consistent ? tf::tet_mesh_status::meshed
                                            : tf::tet_mesh_status::unwound)
                      : tf::tet_mesh_status::refused;
  if (!regions.consistent) {
    if constexpr (reports)
      report.input_open_edges =
          count_tet_open_edges(polygons.faces(), steps).unbalanced;
    return finish();
  }
  phase.segment.unrestored +=
      restore_tet_kept_cells(owner, covering, regions, phase.segment);

  account();
  gather_tet_face_owners(owner, face_owners);
  tf::buffer<char> used;
  used.allocate(dt3::tet_site_count(owner));
  tf::parallel_fill(used, char(0));
  tf::parallel_for_each(
      face_owners,
      [&used, &recovery](const auto &record) {
        if (recovery.status[std::size_t(record.face)] ==
            dt3::tet_face_status::recovered)
          for (auto site : record.key)
            used[std::size_t(site)] = 1;
      },
      tf::checked);
  tf::buffer<Index> point_of;
  mesh = emit_tet_mesh<RealOut, Index>(
      owner, [&regions](std::size_t cell) { return regions.winding[cell] > 0; },
      used, surface.converter, entrance.frame_bits, point_of);
  if constexpr (reports) {
    report.inverted = count_inverted_tets(mesh);
    report.flat = count_flat_tet_cells(owner, [&regions](std::size_t cell) {
      return regions.winding[cell] > 0;
    });
    report.negative_cells = count_negative_tet_cells(regions);
  }

  if constexpr (std::is_same_v<ReturnTag, tf::return_index_map_t>) {
    tet_face_map_workspace<Index> map_work;
    emit_tet_face_children<Index>(owner, facets, recovery.status, point_of,
                                  face_owners, map_work, map.facet_triangles);
    if (report.drains) {
      tf::buffer<Index> kept_facets, kept_parents;
      tf::offset_block_buffer<Index, face_t> kept_children;
      compact_tet_facets(surface.facets, surface.parents, recovery.status,
                         map.facet_triangles, kept_facets, kept_parents,
                         kept_children);
      map.facet_triangles = std::move(kept_children);
      state_tet_face_facets(polygons.faces().size(), kept_facets, kept_parents,
                            map_work, map.face_facets);
    } else
      state_tet_face_facets(polygons.faces().size(), surface.facets,
                            surface.parents, map_work, map.face_facets);
  }
  return finish();
}

} // namespace tf::tetmesh
