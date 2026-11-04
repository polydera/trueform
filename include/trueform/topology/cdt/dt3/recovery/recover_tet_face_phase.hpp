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
#include "../../../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../../../../core/views/offset_block_range.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../core/views/zip.hpp"
#include "../../../../exact/meta.hpp"
#include "../../../../exact/orient3d.hpp"
#include "../../../../exact/projection_axes.hpp"
#include "../check_tetrahedralization_validity.hpp"
#include "../compact_tetrahedralization.hpp"
#include "../find_tet_edge.hpp"
#include "../hold_tet_facet.hpp"
#include "../record_tet_site_cells.hpp"
#include "../remove_tet_edge_by_flips.hpp"
#include "../tet_decision_sites.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_flip_constraints.hpp"
#include "../tet_owner_complex.hpp"
#include "../tet_position.hpp"
#include "../tet_site_position.hpp"
#include "../tet_with_sites.hpp"
#include "./append_deep_tet_face_point.hpp"
#include "./append_tet_face_point.hpp"
#include "./check_tet_face_fixpoint.hpp"
#include "./check_tet_pieces_stand.hpp"
#include "./read_tet_face_cover.hpp"
#include "./restore_tet_delaunay.hpp"
#include "./tet_face_census.hpp"
#include "./tet_face_lattice_point.hpp"
#include "./tet_face_recovery.hpp"
#include "./tet_face_status.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_pierce_in_facet.hpp"
#include "./tet_plane_normal.hpp"
#include "./tet_projected_area2.hpp"
#include "./tet_projected_point.hpp"
#include "./tet_rim_piece.hpp"
#include "./tet_segment_phase.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Recover every face of `faces` in the complex `owner`, its rim recovered
/// by phase S (`phase`): a face is one planar polygon, and it is recovered
/// when the facets of the complex on its plane inside its rim tile it, read
/// by area; which facets they are is the complex's to say. First every
/// face's cover is read in parallel, each face a read of the cells about
/// its rim, each piece on its holder triangle's side (@ref tet_rim_piece),
/// and across its cover's edges (@ref read_tet_face_cover), and the
/// lane is stated from the tickets found; a face whose cover tiles it is
/// recovered there. The faces left take passes in order: an edge of the
/// cover's frontier is read for the cell beyond it whose apex and oppo
/// straddle the plane with their edge piercing the face; that edge is
/// removed by flips under the face at link levels one to three
/// (@ref remove_tet_edge_by_flips), and where flips fail a point is placed
/// exactly on the plane, in the face off its rim, near the pierce
/// (@ref tet_face_lattice_point, @ref append_tet_face_point), so the face
/// stays a plane and no edge from the point pierces it; the cover is read
/// again and grows from the lane's walls and the point's star. Where the
/// face's lattice names no point on the frame, or a pass progresses no edge
/// and a site holds a point it named — the frame's depth witness — the point
/// is placed below the frame, at the shallowest depth whose point lies in
/// the kernel of the piercing edge's ring (@ref append_deep_tet_face_point).
/// A face whose rim segment was refused, that a segment pierces, whose point
/// has no place to the deep lattice's cap or passes the site budget, or
/// whose piercing edges neither flips nor a point removed, is refused; a
/// facet other faces hold already joins their stack. Piercing edges are
/// never made — flips refuse them and points only remove cells — so the
/// passes end. A face's corners are build sites on the frame, so a decision
/// over it reads at the depth of its other corners, its cover at the
/// owner's deepest (@ref tet_with_sites). The complex is then restored to
/// Delaunay where flips reach (@ref restore_tet_delaunay), compacted and
/// published. Returns false only when an append refused, which publishes no
/// cells.
template <bool Parallel, typename Owner, typename Work, typename Faces>
auto recover_tet_face_phase(
    Owner &owner, Work &work, const Faces &faces,
    tet_segment_phase<typename Owner::index_type, typename Owner::coord_type>
        &phase,
    tet_face_recovery<typename Owner::index_type, typename Owner::coord_type,
                      typename Owner::int_type> &recovery) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  using ref_t = tet_edge_ref<Index>;
  auto &segment = phase.segment;
  auto &scratch = segment.flips;
  auto &census = recovery.census;
  auto &status = recovery.status;
  const std::size_t n = faces.size();
  record_tet_site_cells(owner);
  census = tet_face_census{};
  status.allocate(n);
  tf::parallel_fill(status, tet_face_status::pending);
  recovery.points.clear();
  recovery.uncovered.clear();
  scratch.counts.clear();
  scratch.arena.clear();
  scratch.collected.clear();
  scratch.encroached.clear();
  scratch.level_limit = -1;
  scratch.auto_level = 1;
  scratch.star_limit = -1;
  owner._written.clear();
  owner._record_written = true;
  tet_owner_complex<Parallel, Owner, Work> dt(owner, work);
  const auto is_segment = [&segment](Index p, Index q) {
    const Index low = std::min(p, q), high = std::max(p, q);
    const auto &heads = segment.row_heads;
    if (std::size_t(low) >= heads.size())
      return false;
    for (Index t = heads[std::size_t(low)]; t != Index(-1);
         t = segment.row_tickets[std::size_t(t)].second) {
      const auto &row = segment.subsegments[std::size_t(
          segment.row_tickets[std::size_t(t)].first)];
      if ((row.a == low && row.b == high) || (row.a == high && row.b == low))
        return true;
    }
    return false;
  };
  const auto refuse = [&status, &census](Index f, std::size_t &kind) {
    status[std::size_t(f)] = tet_face_status::refused;
    ++census.refused;
    ++kind;
  };
  for (std::size_t f = 0; f < n; ++f)
    for (auto s : phase.edges[f])
      if (segment.status[std::size_t(s)] != tet_segment_status::recovered) {
        refuse(Index(f), census.open_rim);
        break;
      }
  const auto state_rim = [&owner, &phase, &segment](
                             std::size_t f, const std::pair<int, int> &axes,
                             tf::buffer<tet_rim_piece<Index>> &rim) {
    const auto sites = tet_shallow_decision_of(owner);
    const auto flat = [&sites, &axes](Index s) {
      return tet_projected_point(sites[std::size_t(s)].pt, axes);
    };
    rim.clear();
    for (auto &&[s, holder] : tf::make_offset_block_range(
             phase.edges.offsets_buffer(),
             tf::zip(phase.edges.data_buffer(), phase.holders))[f]) {
      const auto chain = segment.chains[std::size_t(s)];
      const int winds = tet_orient2d_sign<Int>(flat(holder[0]), flat(holder[1]),
                                               flat(holder[2]));
      if constexpr (tf::core::verification)
        tf::core::verify(
            winds && ((chain[0].site == holder[0] &&
                       chain[chain.size() - 1].site == holder[1]) ||
                      (chain[0].site == holder[1] &&
                       chain[chain.size() - 1].site == holder[0])),
            "a rim segment's chain runs between the ends its holder winds");
      const int side = chain[0].site == holder[0] ? winds : -winds;
      for (std::size_t i = 1; i < chain.size(); ++i) {
        const Index p = chain[i - 1].site, q = chain[i].site;
        rim.push_back(p < q ? tet_rim_piece<Index>{p, q, side}
                            : tet_rim_piece<Index>{q, p, -side});
      }
    }
    std::sort(rim.begin(), rim.end());
  };
  const auto axes_of = [&owner](const auto &facet) {
    const auto &t = facet[0];
    return tf::exact::projection_axes<Int>(
        tet_plane_normal<Int>(tet_shallow_point(owner, Index(t[0])),
                              tet_shallow_point(owner, Index(t[1])),
                              tet_shallow_point(owner, Index(t[2]))));
  };
  const auto area_of = [](const auto &sites, const auto &facet,
                          const std::pair<int, int> &axes) {
    using lattice = typename std::decay_t<decltype(sites)>::int_type;
    typename tf::exact::meta<lattice>::T2 area(0);
    for (const auto &t : facet)
      area = area + tet_projected_area2(sites, axes, Index(t[0]), Index(t[1]),
                                        Index(t[2]));
    return area;
  };

  struct local_t {
    tf::buffer<Index> sizes;
    tf::buffer<Index> tickets;
    tf::buffer<char> covered;
    tf::buffer<tet_rim_piece<Index>> rim;
    tf::buffer<Index> found;
    tf::buffer<std::array<Index, 2>> frontier;
  };
  auto &cover = recovery.cover;
  cover.offsets_buffer().allocate(n + 1);
  cover.offsets_buffer()[0] = 0;
  cover.data_buffer().clear();
  recovery.covered.allocate(n);
  std::size_t at = 1;
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(n), std::tie(cover, recovery.covered), local_t{},
      [&](auto &&range, local_t &local) {
        local.sizes.clear();
        local.tickets.clear();
        local.covered.clear();
        for (auto f : range) {
          if (status[std::size_t(f)] != tet_face_status::pending) {
            local.sizes.push_back(0);
            local.covered.push_back(0);
            continue;
          }
          const auto facet = faces[std::size_t(f)];
          const auto axes = axes_of(facet);
          state_rim(std::size_t(f), axes, local.rim);
          const bool tiled = tet_with_sites(
              owner, tet_deepest_depth(owner), [&](const auto &sites) {
                return read_tet_face_cover(
                           owner, sites, facet[0], axes, local.rim, local.found,
                           local.frontier) == area_of(sites, facet, axes);
              });
          local.sizes.push_back(Index(local.found.size()));
          for (auto ticket : local.found)
            local.tickets.push_back(ticket);
          local.covered.push_back(tiled);
        }
      },
      [&at](const local_t &local, auto &result) {
        auto &blocks = std::get<0>(result);
        auto &covered = std::get<1>(result);
        for (auto ticket : local.tickets)
          blocks.data_buffer().push_back(ticket);
        auto &offsets = blocks.offsets_buffer();
        for (std::size_t i = 0; i < local.sizes.size(); ++i) {
          offsets[at] = offsets[at - 1] + local.sizes[i];
          covered[at - 1] = local.covered[i];
          ++at;
        }
      });
  for (std::size_t f = 0; f < n; ++f) {
    if (status[f] != tet_face_status::pending)
      continue;
    for (auto ticket : cover[f])
      hold_tet_facet(owner, ticket, Index(f));
    census.covered += cover[f].size();
    if (recovery.covered[f])
      status[f] = tet_face_status::recovered;
    else
      recovery.uncovered.push_back(Index(f));
  }

  bool appended = true;
  tet_flip_constraints<Index, typename Owner::coord_type, Int> fc;
  for (auto f : recovery.uncovered) {
    const auto facet = faces[std::size_t(f)];
    const auto axes = axes_of(facet);
    const T2 full = area_of(tet_shallow_decision_of(owner), facet, axes);
    const auto &first = facet[0];
    const auto pierces = [&owner, &facet, &first, &axes](Index x, Index y) {
      return tet_with_sites(
          owner, tet_deepest_of(owner, x, y), [&](const auto &sites) {
            using lattice = typename std::decay_t<decltype(sites)>::int_type;
            using V2 = typename tf::exact::meta<lattice>::T2;
            const auto &a = sites[std::size_t(first[0])].pt;
            const auto &b = sites[std::size_t(first[1])].pt;
            const auto &c = sites[std::size_t(first[2])].pt;
            const V2 ox = tf::exact::orient3d_value_scaled<lattice>(
                a, b, c, sites[std::size_t(x)].pt);
            const V2 oy = tf::exact::orient3d_value_scaled<lattice>(
                a, b, c, sites[std::size_t(y)].pt);
            return ((ox > V2(0) && oy < V2(0)) || (ox < V2(0) && oy > V2(0))) &&
                   tet_pierce_in_facet(sites, facet, axes, x, y, ox, oy);
          });
    };
    fc.polygon = {&*facet.begin(), &*facet.begin() + facet.size()};
    fc.axes = axes;
    state_rim(std::size_t(f), axes, recovery.rim);
    for (;;) {
      ++census.passes;
      const bool tiled = tet_with_sites(
          owner, tet_deepest_depth(owner), [&](const auto &sites) {
            const auto area =
                read_tet_face_cover(owner, sites, facet[0], axes, recovery.rim,
                                    recovery.tickets, recovery.frontier);
            using V2 = std::decay_t<decltype(area)>;
            if constexpr (std::decay_t<decltype(sites)>::deep)
              return area == V2(full) * (V2(1) << (2 * sites.depth));
            else
              return area == full;
          });
      for (auto ticket : recovery.tickets)
        hold_tet_facet(owner, ticket, f);
      if (tiled) {
        status[std::size_t(f)] = tet_face_status::recovered;
        break;
      }
      bool progressed = false;
      std::size_t *kind = nullptr;
      recovery.standing.clear();
      for (const auto &edge : recovery.frontier) {
        ref_t handle{};
        if (!find_tet_edge(owner, edge[0], edge[1], handle))
          continue;
        auto spin = handle;
        Index d = Owner::none, e = Owner::none;
        do {
          const auto c = tet_edge_corners(owner, spin);
          if (c[2] != Owner::infinite && c[3] != Owner::infinite &&
              pierces(c[2], c[3])) {
            d = c[2];
            e = c[3];
            break;
          }
          spin = tet_edge_next(owner, spin);
        } while (spin.cell != handle.cell);
        if (d == Owner::none)
          continue;
        if (is_segment(d, e)) {
          kind = &census.crossed;
          break;
        }
        bool removed = false;
        for (int level = 1; level <= 3 && !removed; ++level) {
          ref_t pierce{};
          if (!find_tet_edge(owner, d, e, pierce))
            break;
          scratch.auto_level = level;
          removed = remove_tet_edge_by_flips(owner, scratch, pierce, fc,
                                             is_segment) == 2;
        }
        if (removed) {
          ++census.flipped;
          progressed = true;
          break;
        }
        const Index hint = owner._site_cells.load(std::size_t(edge[0]));
        tet_position<typename Owner::coord_type> point{};
        Index site = Owner::none;
        const bool framed =
            tet_face_lattice_point(owner, facet, axes, d, e, 0, point);
        const auto route =
            framed ? append_tet_face_point(dt, recovery, segment, f, point,
                                           hint, is_segment, site)
                   : append_deep_tet_face_point(dt, recovery, segment, f, facet,
                                                axes, d, e, hint, is_segment,
                                                site);
        if (route == tet_face_point_route::placed) {
          ++census.points;
          progressed = true;
          break;
        }
        if (route == tet_face_point_route::budget) {
          kind = &census.budget;
          break;
        }
        if (route == tet_face_point_route::failed) {
          appended = false;
          break;
        }
        if (!framed) {
          kind = &census.off_lattice;
          break;
        }
        if (route == tet_face_point_route::standing)
          recovery.standing.push_back({d, e, hint});
      }
      for (std::size_t i = 0;
           appended && !kind && !progressed && i < recovery.standing.size();
           ++i) {
        const auto &at = recovery.standing[i];
        Index site = Owner::none;
        const auto route =
            append_deep_tet_face_point(dt, recovery, segment, f, facet, axes,
                                       at[0], at[1], at[2], is_segment, site);
        if (route == tet_face_point_route::placed) {
          ++census.points;
          progressed = true;
        } else if (route == tet_face_point_route::budget)
          kind = &census.budget;
        else if (route == tet_face_point_route::failed)
          appended = false;
      }
      if (!appended)
        break;
      if (kind) {
        refuse(f, *kind);
        break;
      }
      if (!progressed) {
        refuse(f, census.pierced);
        census.standing += recovery.standing.size() != 0;
        break;
      }
    }
    if (!appended)
      break;
  }
  if (appended)
    segment.unrestored += restore_tet_delaunay(owner, segment);
  owner._record_written = false;
  owner._written.clear();
  compact_tetrahedralization<true, true>(owner);
  std::sort(recovery.points.begin(), recovery.points.end());
  if constexpr (tf::core::verification) {
    tf::core::verify(!appended || check_tetrahedralization_validity(owner),
                     "the face phase publishes a valid tetrahedralization");
    tf::core::verify(
        !appended || check_tet_face_fixpoint(owner, faces, recovery),
        "every face recovered is tiled by the lane facets it owns");
    tf::core::verify(!appended || check_tet_pieces_stand(owner, phase.pieces),
                     "every recovered segment piece stands as an edge");
  }
  return appended;
}

} // namespace tf::topology::cdt::dt3
