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
#include "../../../../core/algorithm/generic_generate.hpp"
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../../../../core/views/offset_block_range.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../core/views/zip.hpp"
#include "../../../../exact/meta.hpp"
#include "../../../../exact/orient3d.hpp"
#include "../../../../exact/projection_axes.hpp"
#include "../find_tet_edge.hpp"
#include "../hold_tet_facet.hpp"
#include "./regions/run_tet_region_round.hpp"
#include "../mark_tet_written_rows.hpp"
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
#include "./advance_tet_face_cover.hpp"
#include "./append_deep_tet_face_point.hpp"
#include "./for_each_tet_chain_piece.hpp"
#include "./append_tet_face_point.hpp"
#include "./check_tet_face_cover.hpp"
#include "./check_tet_face_fixpoint.hpp"
#include "./check_tet_segment_fixpoint.hpp"
#include "./find_tet_face_facet.hpp"
#include "./read_tet_face_cover.hpp"
#include "./state_tet_site_cells.hpp"
#include "./tet_crossing_pair.hpp"
#include "./tet_edge_facet_owner.hpp"
#include "./tet_face_census.hpp"
#include "./tet_face_crossing.hpp"
#include "./tet_face_holds_facet.hpp"
#include "./tet_face_lattice_point.hpp"
#include "./tet_face_recovery.hpp"
#include "./tet_face_status.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_piece_segment.hpp"
#include "./tet_pierce_in_facet.hpp"
#include "./tet_plane_normal.hpp"
#include "./tet_projected_point.hpp"
#include "./tet_refusing_cut.hpp"
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
/// when the facets of the complex on its plane inside its rim tile it, which
/// they do exactly when its cover leaves no frontier; which facets they are
/// is the complex's to say. First every
/// face's cover is stated in parallel: a face of one triangle that stands
/// as a facet is covered by that facet alone (@ref find_tet_face_facet),
/// and every other face's cover is read from the cells about its rim, each
/// piece on its holder triangle's side (@ref tet_rim_piece), and across its
/// cover's edges (@ref read_tet_face_cover); the lane is stated from the
/// tickets found, and a face whose cover tiles it is recovered there. The
/// faces left take passes in order: an edge of the
/// cover's frontier is read for the cell beyond it whose apex and oppo
/// straddle the plane with their edge piercing the face; that edge is
/// removed by flips under the face at link levels one to three
/// (@ref remove_tet_edge_by_flips), and where flips fail a point is placed
/// exactly on the plane, in the face off its rim, near the pierce
/// (@ref tet_face_lattice_point, @ref append_tet_face_point), so the face
/// stays a plane and no edge from the point pierces it; the cover then
/// advances from the frontier the rows written since name and the piece
/// the point divided (@ref advance_tet_face_cover), and is read again from
/// the rim only after a compaction renumbered those rows. Where the
/// face's lattice names no point on the frame, or a pass progresses no edge
/// and a site holds a point it named — the frame's depth witness — the point
/// is placed below the frame, at the shallowest depth whose point lies in
/// the kernel of the piercing edge's ring (@ref append_deep_tet_face_point).
/// That search reads the edge's ring and the sites standing in it alone, so
/// an edge it found no point for is not searched again on the face until a
/// row holding the edge is written — the edges the cover's advance names —
/// or a compaction renumbers the rows. A face whose rim segment was refused, that a segment pierces, whose point
/// has no place to the deep lattice's cap or passes the site budget, or
/// whose piercing edges neither flips nor a point removed, is refused; a
/// facet other faces hold already joins their stack. A face a crossing
/// stops names what it crosses (@ref tet_face_crossing): the segment whose
/// piece pierces it, or, where its piercing edges neither flip nor a point
/// removed, the face whose protected facet holds one of them
/// (@ref tet_edge_facet_owner), and hands it to `cut` (@ref
/// tet_crossing_pair); the faces a cut made due, stated into
/// `phase.segment.due`, are walked in the same pass, and a crossing it does
/// not cut refuses the face. Piercing edges are
/// never made — flips refuse them and points only remove cells — so the
/// passes end. A face's corners are build sites on the frame, so a decision
/// over it reads at the depth of its other corners, its cover at the
/// owner's deepest (@ref tet_with_sites). The rows stay as they stand, the
/// publish being the caller's barrier (@ref compact_tetrahedralization), and
/// the rows phases S and F wrote stay recorded for the restoration after the
/// region labeling (@ref restore_tet_delaunay). A FACE THE ADMISSION STOPPED
/// IS THE FRONTIER'S and never a refusal: its walk made no progress only
/// because a row of another region stood in it, so it defers
/// (@ref admit_tet_row). Returns false only when an
/// append refused.
///
/// The phase is a diff on the faces `recovery` holds: only the faces `due`
/// names, ascending, are recovered — each due again, its status and crossing
/// stated afresh — and every other face keeps its status, its cover and the
/// facets it holds; the census adds this call's routes to the standing
/// ones. The stock call names every face of `faces`.
template <bool Parallel, typename Owner, typename Work, typename Faces,
          typename Due, typename Cut = tet_refusing_cut>
auto recover_tet_face_phase(
    Owner &owner, Work &work, const Faces &faces, const Due &due,
    tet_segment_phase<typename Owner::index_type, typename Owner::coord_type>
        &phase,
    tet_face_recovery<typename Owner::index_type, typename Owner::coord_type,
                      typename Owner::int_type> &recovery,
    const Cut &cut = Cut{}) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  using ref_t = tet_edge_ref<Index>;
  auto &segment = phase.segment;
  auto &scratch = segment.flips;
  auto &census = recovery.census;
  auto &status = recovery.status;
  const std::size_t n = faces.size();
  const std::size_t n_due = due.size();
  record_tet_site_cells(owner);
  status.reallocate(n);
  recovery.crossing.reallocate(n);
  recovery.deep.reallocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n_due),
      [&status, &recovery, &due](std::size_t i) {
        status[std::size_t(due[i])] = tet_face_status::pending;
        recovery.crossing[std::size_t(due[i])] = tet_face_crossing<Index>{};
        recovery.deep[std::size_t(due[i])] = 0;
      },
      tf::checked);
  recovery.uncovered.clear();
  scratch.touched.clear();
  scratch.arena.clear();
  scratch.collected.clear();
  scratch.level_limit = -1;
  scratch.auto_level = 1;
  scratch.star_limit = -1;
  owner._record_written = true;
  tet_owner_complex<Parallel, Owner, Work> dt(owner, work);
  const auto is_segment = [&segment](Index p, Index q) {
    return tet_piece_segment(segment, p, q) != Index(-1);
  };
  const auto refuse = [&status, &census, &recovery](Index f,
                                                    std::size_t &kind) {
    status[std::size_t(f)] = tet_face_status::refused;
    recovery.deep[std::size_t(f)] = &kind == &census.off_lattice;
    ++census.refused;
    ++kind;
  };
  const auto settle = [&status, &recovery, &scratch,
                       &refuse](Index f, std::size_t &kind) {
    if (!scratch.round.stopped) {
      refuse(f, kind);
      return;
    }
    status[std::size_t(f)] = tet_face_status::pending;
    recovery.deferred.push_back(f);
  };
  for (std::size_t i = 0; i < n_due; ++i)
    for (auto s : phase.edges[std::size_t(phase.edge_block[std::size_t(due[i])])])
      if (segment.status[std::size_t(s)] != tet_segment_status::recovered) {
        refuse(Index(due[i]), census.open_rim);
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
      const auto &rows = segment.subsegments;
      const Index head = rows[std::size_t(segment.first[std::size_t(s)])].a;
      const Index tail = rows[std::size_t(segment.last[std::size_t(s)])].b;
      const int winds = tet_orient2d_sign<Int>(flat(holder[0]), flat(holder[1]),
                                               flat(holder[2]));
      if constexpr (tf::core::verification)
        tf::core::verify(
            winds && ((head == holder[0] && tail == holder[1]) ||
                      (head == holder[1] && tail == holder[0])),
            "a rim segment's chain runs between the ends its holder winds");
      const int side = head == holder[0] ? winds : -winds;
      for_each_tet_chain_piece(
          segment, std::size_t(s), segment.bits[std::size_t(s)],
          [&rim, side](Index p, Index q, std::int64_t, std::int64_t) {
            rim.push_back(p < q ? tet_rim_piece<Index>{p, q, side}
                                : tet_rim_piece<Index>{q, p, -side});
          });
    }
    std::sort(rim.begin(), rim.end(), tet_rim_piece_precedes<Index>);
  };
  const auto axes_of = [&owner](const auto &facet) {
    const auto &t = facet[0];
    return tf::exact::projection_axes<Int>(
        tet_plane_normal<Int>(tet_shallow_point(owner, Index(t[0])),
                              tet_shallow_point(owner, Index(t[1])),
                              tet_shallow_point(owner, Index(t[2]))));
  };

  struct local_t {
    tf::buffer<Index> sizes;
    tf::buffer<Index> tickets;
    tf::buffer<Index> layers;
    tf::buffer<Index> read_layers;
    tf::buffer<char> covered;
    tet_face_cover_scratch<Index> cover;
  };
  auto &cover = recovery.cover;
  cover.offsets_buffer().allocate(n_due + 1);
  cover.offsets_buffer()[0] = 0;
  cover.data_buffer().clear();
  recovery.cover_layers.clear();
  recovery.covered.allocate(n_due);
  auto &star_sites = recovery.star_sites;
  star_sites.clear();
  tf::generic_generate(
      due, star_sites,
      [&faces, &status](auto f, auto &out) {
        const auto facet = faces[std::size_t(f)];
        if (status[std::size_t(f)] == tet_face_status::pending &&
            facet.size() == 1)
          for (std::size_t k = 0; k < 3; ++k)
            out.push_back(Index(facet[0][k]));
      },
      tf::checked);
  state_tet_site_cells(dt, star_sites, recovery.site_cells,
                       recovery.site_scratch);
  std::size_t at = 1;
  tf::blocked_reduce_sequenced_aggregate(
      due, std::tie(cover, recovery.covered), local_t{},
      [&](auto &&range, local_t &local) {
        local.sizes.clear();
        local.tickets.clear();
        local.layers.clear();
        local.covered.clear();
        for (auto f : range) {
          if (status[std::size_t(f)] != tet_face_status::pending) {
            local.sizes.push_back(0);
            local.covered.push_back(0);
            continue;
          }
          const auto facet = faces[std::size_t(f)];
          Index ticket = Owner::none;
          if (facet.size() == 1 &&
              find_tet_face_facet(owner, recovery.star_sites,
                                  recovery.site_cells, facet[0],
                                  ticket)) {
            if constexpr (tf::core::verification) {
              const auto axes = axes_of(facet);
              state_rim(std::size_t(f), axes, local.cover.rim);
              tet_with_sites(
                  owner, tet_deepest_depth(owner), [&](const auto &sites) {
                    read_tet_face_cover(owner, sites, facet[0], axes,
                                        local.cover.rim, local.cover.tickets,
                                        local.cover.frontier, local.read_layers,
                                        local.cover);
                  });
              tf::core::verify(
                  !local.cover.frontier.size() &&
                      local.cover.tickets.size() == 1 &&
                      local.cover.tickets[0] == ticket &&
                      check_tet_face_cover(owner, facet, axes, local.cover.rim,
                                           local.cover.frontier, Owner::none,
                                           local.cover.check_tickets,
                                           local.cover.check_frontier,
                                           local.cover),
                  "a face standing as a facet is covered by that facet alone");
            }
            local.sizes.push_back(1);
            local.tickets.push_back(ticket);
            local.layers.push_back(1);
            local.covered.push_back(1);
            continue;
          }
          const auto axes = axes_of(facet);
          state_rim(std::size_t(f), axes, local.cover.rim);
          tet_with_sites(
              owner, tet_deepest_depth(owner), [&](const auto &sites) {
                read_tet_face_cover(owner, sites, facet[0], axes,
                                    local.cover.rim, local.cover.tickets,
                                    local.cover.frontier, local.read_layers,
                                    local.cover);
              });
          if constexpr (tf::core::verification)
            tf::core::verify(
                check_tet_face_cover(owner, facet, axes, local.cover.rim,
                                     local.cover.frontier, Owner::none,
                                     local.cover.check_tickets,
                                     local.cover.check_frontier, local.cover),
                "a face is tiled exactly when its cover leaves "
                "no frontier");
          local.sizes.push_back(Index(local.cover.tickets.size()));
          for (std::size_t i = 0; i < local.cover.tickets.size(); ++i) {
            local.tickets.push_back(local.cover.tickets[i]);
            local.layers.push_back(
                local.read_layers.size() ? local.read_layers[i] : Index(1));
          }
          local.covered.push_back(!local.cover.frontier.size());
        }
      },
      [&at, &recovery](const local_t &local, auto &result) {
        auto &blocks = std::get<0>(result);
        auto &covered = std::get<1>(result);
        for (auto ticket : local.tickets)
          blocks.data_buffer().push_back(ticket);
        for (auto layers : local.layers)
          recovery.cover_layers.push_back(layers);
        auto &offsets = blocks.offsets_buffer();
        for (std::size_t i = 0; i < local.sizes.size(); ++i) {
          offsets[at] = offsets[at - 1] + local.sizes[i];
          covered[at - 1] = local.covered[i];
          ++at;
        }
      });
  // THE HOLDS ARE A ROUND: one task is one cover entry, anchored at the row
  // whose facet it holds, and an entry the admission stopped is held by the
  // serial pass that follows.
  {
    auto &holder = recovery.hold_face;
    const std::size_t entries = cover.data_buffer().size();
    holder.allocate(entries);
    tf::parallel_for_each(
        tf::make_sequence_range(n_due),
        [&holder, &cover, &due, &status](std::size_t i) {
          const auto f = due[i];
          const bool pending =
              status[std::size_t(f)] == tet_face_status::pending;
          for (auto at = cover.offsets_buffer()[i];
               at < cover.offsets_buffer()[i + 1]; ++at)
            holder[std::size_t(at)] = pending ? Index(f) : Index(-1);
        },
        tf::checked);
    const auto hold_anchor = [&owner, &cover, &segment](std::size_t at)
        -> Index {
      const Index cell =
          Index(std::size_t(cover.data_buffer()[at]) / 4);
      const Index site = owner._corners[std::size_t(cell)][0];
      return site >= 0 && std::size_t(site) < segment.site_patches.size()
                 ? segment.site_patches[std::size_t(site)]
                 : Index(-1);
    };
    const auto hold_task = [&](tet_flip_scratch<Index> &own, Index at) {
      const Index f = holder[std::size_t(at)];
      if (f < 0)
        return true;
      hold_tet_facet(owner, cover.data_buffer()[std::size_t(at)], f,
                     recovery.cover_layers[std::size_t(at)], own.round);
      return !own.round.stopped;
    };
    if (run_tet_region_round(owner,
                             tet_standing_partition{segment.n_regions},
                             segment.site_patches, entries, 0, hold_anchor,
                             scratch, recovery.round_work, hold_task,
                             recovery.region_left, scratch)) {
      recovery.region_tasks += entries;
      recovery.region_deferred += recovery.region_left.size();
      for (auto at : recovery.region_left)
        if (holder[std::size_t(at)] >= 0)
          hold_tet_facet(owner, cover.data_buffer()[std::size_t(at)],
                         holder[std::size_t(at)],
                         recovery.cover_layers[std::size_t(at)]);
    } else
      for (std::size_t at = 0; at < entries; ++at)
        if (holder[std::size_t(at)] >= 0)
          hold_tet_facet(owner, cover.data_buffer()[std::size_t(at)],
                         holder[std::size_t(at)],
                         recovery.cover_layers[std::size_t(at)]);
  }
  for (std::size_t i = 0; i < n_due; ++i) {
    const auto f = std::size_t(due[i]);
    if (status[f] != tet_face_status::pending)
      continue;
    census.covered += cover[i].size();
    if (recovery.covered[i])
      status[f] = tet_face_status::recovered;
    else
      recovery.uncovered.push_back(Index(f));
  }

  bool appended = true;
  tet_flip_constraints<Index, typename Owner::coord_type, Int> fc;
  const auto cuts = [&](Index f) {
    auto &made = segment.due;
    made.rows.clear();
    made.faces.clear();
    if (!cut(tet_crossing_pair<Index>{Index(-1), f,
                                      recovery.crossing[std::size_t(f)]},
             made))
      return false;
    for (auto g : made.faces) {
      status[std::size_t(g)] = tet_face_status::pending;
      recovery.crossing[std::size_t(g)] = tet_face_crossing<Index>{};
      recovery.deep[std::size_t(g)] = 0;
      recovery.uncovered.push_back(g);
    }
    return true;
  };
  const auto walk_face = [&](Index f) {
    const auto facet = faces[std::size_t(f)];
    const auto axes = axes_of(facet);
    const auto &first = facet[0];
    const auto pierces = [&owner, &facet, &first, &axes](Index x, Index y) {
      return tet_with_sites(
          owner, tet_deepest_of(owner, x, y), [&](const auto &sites) {
            using lattice = typename std::decay_t<decltype(sites)>::int_type;
            using V2 = typename tf::exact::meta<lattice>::T2;
            const auto a = sites[std::size_t(first[0])].pt;
            const auto b = sites[std::size_t(first[1])].pt;
            const auto c = sites[std::size_t(first[2])].pt;
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
    state_rim(std::size_t(f), axes, recovery.cover_scratch.rim);
    recovery.exhausted.clear();
    bool read = false;
    std::size_t mark = 0, epoch = 0;
    std::array<Index, 3> divided{Owner::none, Owner::none, Owner::none};
    std::size_t passes = 0;
    for (;;) {
      ++census.passes;
      // the chain's own bound: a pass reads the cover, flips or places, and a
      // chain that spends its passes refuses rather than running on
      if (passes++ >= recovery.chain_passes) {
        settle(f, census.capped);
        break;
      }
      const bool advance = read && epoch == owner._compaction_epoch;
      tet_with_sites(owner, tet_deepest_depth(owner), [&](const auto &sites) {
        if (advance)
          advance_tet_face_cover(
              owner, sites, facet[0], axes, mark, divided, recovery.touched,
              recovery.seeds, recovery.cover_scratch.tickets,
              recovery.cover_scratch.frontier, recovery.cover_scratch);
        else
          read_tet_face_cover(
              owner, sites, facet[0], axes, recovery.cover_scratch.rim,
              recovery.cover_scratch.tickets, recovery.cover_scratch.frontier,
              recovery.layers, recovery.cover_scratch);
      });
      if (advance) {
        const auto &touched = recovery.touched;
        std::size_t kept = 0;
        for (const auto &edge : recovery.exhausted) {
          const std::array<Index, 2> key{std::min(edge[0], edge[1]),
                                         std::max(edge[0], edge[1])};
          if (!std::binary_search(touched.begin(), touched.end(), key))
            recovery.exhausted[kept++] = edge;
        }
        recovery.exhausted.erase_till_end(recovery.exhausted.begin() +
                                          std::ptrdiff_t(kept));
        std::sort(recovery.exhausted.begin(), recovery.exhausted.end());
      } else
        recovery.exhausted.clear();
      const std::size_t searched = recovery.exhausted.size();
      read = true;
      mark = mark_tet_written_rows(owner);
      epoch = owner._compaction_epoch;
      divided = {Owner::none, Owner::none, Owner::none};
      for (auto ticket : recovery.cover_scratch.tickets) {
        if constexpr (tf::core::verification)
          tf::core::verify(!advance || !tet_face_holds_facet(owner, ticket, f),
                           "an advance gains only facets its face does not "
                           "hold");
        hold_tet_facet(owner, ticket, f);
      }
      if constexpr (tf::core::verification)
        tf::core::verify(
            check_tet_face_cover(owner, facet, axes, recovery.cover_scratch.rim,
                                 recovery.cover_scratch.frontier, f,
                                 recovery.cover_scratch.check_tickets,
                                 recovery.cover_scratch.check_frontier,
                                 recovery.cover_scratch),
            "the cover a pass holds is the cover read from the "
            "rim, held by its face");
      if (!recovery.cover_scratch.frontier.size()) {
        // an advance gains facets without the layers over them: the whole
        // cover is read once more to state them
        if (advance && facet.size() > 1)
          tet_with_sites(
              owner, tet_deepest_depth(owner), [&](const auto &sites) {
                read_tet_face_cover(owner, sites, facet[0], axes,
                                    recovery.cover_scratch.rim,
                                    recovery.cover_scratch.check_tickets,
                                    recovery.cover_scratch.check_frontier,
                                    recovery.layers, recovery.cover_scratch);
              });
        const auto &whole = advance && facet.size() > 1
                                ? recovery.cover_scratch.check_tickets
                                : recovery.cover_scratch.tickets;
        for (std::size_t i = 0; i < recovery.layers.size(); ++i)
          if (recovery.layers[i] > 1)
            hold_tet_facet(owner, whole[i], f, recovery.layers[i]);
        status[std::size_t(f)] = tet_face_status::recovered;
        break;
      }
      bool progressed = false;
      bool cut_here = false;
      std::size_t *kind = nullptr;
      recovery.standing.clear();
      recovery.piercing.clear();
      for (const auto &edge : recovery.cover_scratch.frontier) {
        ref_t handle{};
        if (!find_tet_edge(owner, edge.lo, edge.hi, handle,
                           recovery.cover_scratch.star, scratch.round))
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
        const Index crossed = tet_piece_segment(segment, d, e);
        if (crossed != Index(-1)) {
          recovery.crossing[std::size_t(f)].segment = crossed;
          if (cuts(f))
            cut_here = true;
          else
            kind = &census.crossed;
          break;
        }
        recovery.piercing.push_back({d, e});
        bool removed = false;
        for (int level = 1; level <= 3 && !removed; ++level) {
          ref_t pierce{};
          if (!find_tet_edge(owner, d, e, pierce, recovery.cover_scratch.star,
                             scratch.round))
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
        if (scratch.round.stopped)
          break;
        const Index hint = owner._site_cells.load(std::size_t(edge.lo));
        tet_position<typename Owner::coord_type> point{};
        Index site = Owner::none;
        std::array<Index, 2> on{Owner::none, Owner::none};
        const bool framed =
            tet_face_lattice_point(owner, facet, axes, d, e, 0, point);
        const auto route =
            framed ? append_tet_face_point(dt, segment, facet, point, hint,
                                           is_segment, recovery.near_stamps,
                                           recovery.reached, site, on)
                   : append_deep_tet_face_point(dt, recovery, segment, facet,
                                                axes, d, e, hint, is_segment,
                                                site, on);
        if (route == tet_face_point_route::placed) {
          ++census.points;
          progressed = true;
          divided = {on[0], on[1], site};
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
        if (route == tet_face_point_route::unplaced) {
          ++census.unplaced;
          continue;
        }
        if (!framed) {
          kind = &census.off_lattice;
          break;
        }
        if (route == tet_face_point_route::standing)
          recovery.standing.push_back({d, e, hint});
        else
          ++census.on_piece;
      }
      for (std::size_t i = 0;
           appended && !kind && !cut_here && !progressed &&
           i < recovery.standing.size();
           ++i) {
        const auto &at = recovery.standing[i];
        const std::array<Index, 2> edge{at[0], at[1]};
        if (std::binary_search(recovery.exhausted.begin(),
                               recovery.exhausted.begin() +
                                   std::ptrdiff_t(searched),
                               edge))
          continue;
        Index site = Owner::none;
        std::array<Index, 2> on{Owner::none, Owner::none};
        if (scratch.round.stopped)
          break;
        const auto route = append_deep_tet_face_point(
            dt, recovery, segment, facet, axes, at[0], at[1], at[2], is_segment,
            site, on);
        if (route == tet_face_point_route::standing)
          recovery.exhausted.push_back(edge);
        if (route == tet_face_point_route::placed) {
          ++census.points;
          progressed = true;
          divided = {on[0], on[1], site};
        } else if (route == tet_face_point_route::budget)
          kind = &census.budget;
        else if (route == tet_face_point_route::failed)
          appended = false;
        else if (route == tet_face_point_route::unplaced)
          ++census.unplaced;
      }
      if (!appended)
        break;
      if (cut_here)
        break;
      if (kind) {
        settle(f, *kind);
        break;
      }
      if (!progressed) {
        for (std::size_t i = 0; i < recovery.piercing.size() &&
                                recovery.crossing[std::size_t(f)].face < 0;
             ++i)
          recovery.crossing[std::size_t(f)].face = tet_edge_facet_owner(
              owner, recovery.piercing[i][0], recovery.piercing[i][1],
              recovery.cover_scratch.star, scratch.round);
        if (cuts(f))
          break;
        settle(f, census.pierced);
        census.standing += recovery.standing.size() != 0;
        recovery.deep[std::size_t(f)] = recovery.standing.size() != 0;
        break;
      }
    }
  };
  std::size_t walked = 0;
  for (; walked < recovery.uncovered.size(); ++walked) {
    walk_face(recovery.uncovered[walked]);
    if (!appended)
      break;
  }
  if constexpr (tf::core::verification) {
    tf::core::verify(
        !appended || check_tet_face_fixpoint(owner, faces, recovery),
        "every face recovered is tiled by the lane facets it owns");
    tf::core::verify(!appended || check_tet_segment_fixpoint(owner, phase),
                     "every recovered segment piece stands as an edge");
    for (std::size_t i = 0; i < n_due; ++i) {
      const auto f = std::size_t(due[i]);
      tf::core::verify(!appended || status[f] == tet_face_status::refused ||
                           (recovery.crossing[f].segment == Index(-1) &&
                            recovery.crossing[f].face == Index(-1) &&
                            !recovery.deep[f]),
                       "a due face the phase did not refuse names no crossing "
                       "and no deep refusal");
    }
  }
  return appended;
}

} // namespace tf::topology::cdt::dt3
