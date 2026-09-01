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

#include "../../arrangement/planes/make_plane_piece_fences.hpp"
#include "../../arrangement/planes/make_plane_piece_incidence.hpp"
#include "../../arrangement/planes/plane_arrangement.hpp"
#include "../../arrangement/planes/plane_arrangement_face.hpp"
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/det2_sign.hpp"
#include "../../exact/dot_sign.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/plane_edge_radial_authority.hpp"
#include "./make_plane_triangle_faces.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tf::csg::graph {

/// The radial fans of the arrangement: one fan per fan piece, its pages
/// in radial order around the edge.
///
/// A PAGE is one carrier plane on one of its two sides — the half-plane
/// the sectors meet at — and it carries every live occurrence sitting on
/// it, so triangulation multiplicity and stack depth cost the ring
/// nothing. `pieces[k]` is the k-th fan's piece ticket and
/// `[page_offsets[k], page_offsets[k + 1])` its pages, radially sorted;
/// `rows[p]` is page `p`'s occurrence rows (`triangle * 3 + slot`) and
/// `dirs` walks in lockstep with the row data — 1 when the occurrence
/// traverses the edge in its canonical key order. Every fan piece has a
/// fan, whatever its size: a two-page fan closes its curves and divides
/// like any other.
///
/// `n_refused` counts the fans whose own definitions named no single carrier
/// line (@ref tf::intersect::graph::make_plane_edge_radial_authority) — an
/// ill-posed piece, ordered by the pages' own coordinates instead.
template <typename Index> struct plane_radial_fans {
  tf::buffer<Index> pieces;
  tf::buffer<Index> page_offsets;
  tf::offset_block_buffer<Index, Index> rows;
  tf::buffer<char> dirs;
  Index n_refused = 0;
};

/// Is it a fan? Collect all the planes around it.
///
/// The admission is the fence's `fan` verdict and nothing else. A page is
/// its carrier plane's IDENTITY on one side, never a triangle: the
/// occurrences of one half-plane — a stack's members, a plane's several
/// triangles — are one page and pair once.
///
/// THE CARRIER STATES THE HALF-PLANE. A member's winding in its carrier's
/// frame is the sign of its normal against the carrier's, so an
/// occurrence's side is that stored winding turned by the walk direction,
/// and the page's wedge is the carrier's own normal on that side. The
/// ranking therefore stands on the ELECTED plane, the same one the
/// arrangement was computed in; nothing here reconstructs a normal from a
/// face, and nothing elects a reference member.
///
/// The radial order is exact sign predicates around the carrier LINE, and
/// the line is the piece's own — @ref
/// tf::intersect::graph::make_plane_edge_radial_authority reads it off every
/// definition of the piece. Two pages pair the same way in either order, so
/// the overwhelming majority of fans never ask for it. A piece whose
/// definitions name no single line refuses, and only there does the ring come
/// off the pages' coordinates.
template <typename Index, typename Int, typename Immutable,
          typename GetMeshPoint, typename ApplyToFace>
auto make_plane_radial_fans(
    const tf::arrangement::plane_arrangement<Index, Int> &arrangement,
    const Immutable &immutable,
    const tf::arrangement::plane_piece_incidence<Index> &incidence,
    const tf::arrangement::plane_piece_fences &fences,
    const GetMeshPoint &get_mesh_point, const ApplyToFace &apply_to_face)
    -> plane_radial_fans<Index> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using nvec = std::array<T2, 3>;

  plane_radial_fans<Index> out;
  out.page_offsets.push_back(Index(0));
  out.rows.offsets_buffer().push_back(Index(0));
  if (std::none_of(fences.fan.begin(), fences.fan.end(),
                   [](char fan) { return fan != char(0); }))
    return out;

  const auto coplanar_of = arrangement.coplanar_of();
  const auto triangles = arrangement.triangles();
  const auto face_of = make_plane_triangle_faces(arrangement);

  const auto get_base_point = [&](std::int16_t tag, Index id) {
    return tag < std::int16_t(0)
               ? immutable.point_of(id, get_mesh_point)
               : get_mesh_point(int(tag), id);
  };
  const auto base_created = immutable.n_created_points();
  const auto endpoint = [&](std::int16_t tag, Index id) {
    if (tag >= std::int16_t(0))
      return get_mesh_point(int(tag), id);
    if (id < base_created)
      return immutable.point_of(id, get_mesh_point);
    return arrangement.resolve_created_point(id, get_base_point,
                                             get_mesh_point);
  };

  const auto descriptor_of_face =
      [&](Index face) -> const tf::intersect::graph::face_descriptor<Index> & {
    return tf::arrangement::plane_arrangement_face_descriptor(arrangement,
                                                              immutable, face);
  };

  // THE CARRIER ANSWERS EVERY GEOMETRIC QUESTION ABOUT ITS MEMBERS. A
  // member's stored winding IS the sign of its normal against its
  // carrier's, so an occurrence's side is that winding turned by the walk
  // direction, and the page's wedge is the carrier's own normal on that
  // side. Nothing here reconstructs a normal from a face.
  const auto side_of = [&](Index face, char dir) -> signed char {
    const auto orientation =
        tf::arrangement::plane_arrangement_face_orientation(arrangement,
                                                            immutable, face);
    return static_cast<signed char>(dir ? orientation : -orientation);
  };
  const auto wedge_of = [&](Index plane, signed char side) -> nvec {
    const auto &normal = tf::arrangement::plane_arrangement_plane_frame(
                             arrangement, immutable, plane)
                             .plane_n;
    return side > 0 ? normal
                    : nvec{T2(-normal[0]), T2(-normal[1]), T2(-normal[2])};
  };

  const auto sign_t2 = [](const T2 &v) -> int {
    return (v > 0) ? 1 : (v < 0) ? -1 : 0;
  };
  // THE SENSE OF TWO WEDGES A TURN HAS PROVEN PARALLEL: one is a multiple
  // of the other, so the sign of one shared component's product IS the sign
  // of their dot — read on the component that carries them, which a degree
  // four dot cannot be formed to answer.
  const auto same_sense = [&](const nvec &a, const nvec &b) -> int {
    const auto magnitude = [](const T2 &v) { return v < T2(0) ? -v : v; };
    std::size_t c = 0;
    for (std::size_t k = 1; k < 3; ++k)
      if (magnitude(a[k]) > magnitude(a[c]))
        c = k;
    return sign_t2(a[c]) * sign_t2(b[c]);
  };
  const auto cross3 = [](const nvec &a, const nvec &b) -> nvec {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  };
  const auto is_zero = [](const nvec &v) -> bool {
    return v[0] == 0 && v[1] == 0 && v[2] == 0;
  };

  /// One live occurrence: the row that states it, the face it came out
  /// of, and the page it sits on — its carrier plane and the side of it.
  struct occurrence_t {
    Index plane;
    Index face;
    Index row;
    char dir;
    signed char side;
  };
  struct page_t {
    nvec wedge;
    Index plane;
    Index begin;
    Index count;
    signed char side;
  };
  struct local_t {
    tf::buffer<Index> pieces;
    tf::buffer<Index> page_counts;
    tf::buffer<Index> row_counts;
    tf::buffer<Index> rows;
    tf::buffer<char> dirs;
    tf::buffer<occurrence_t> occurrences;
    tf::buffer<page_t> pages;
    Index refusals = 0;
  };

  // THE ILL-POSED PIECE'S RING. Several carrier lines welded into one
  // canonical identity leave the piece with no line of its own, so the turn
  // is taken from the pages themselves: the cross of the first two
  // independent wedges, turned to the order the resolved endpoints stand in.
  //
  // The cross of two wedges is degree four, which has no rung on the
  // ladder, and the dot against it reads its operands past the width its
  // own contract states: on a carrier standing at the lattice's full span
  // both leave their type. A ring read off the piece's own endpoints is
  // exact and needs neither, and it is NOT what this states — it moves
  // corpus pairs in both directions, so it is an open question, not a fix.
  const auto pages_ring =
      [&](const tf::buffer<page_t> &pages, const auto &reference)
      -> tf::intersect::graph::plane_edge_radial_authority {
    tf::intersect::graph::plane_edge_radial_authority ring;
    nvec d{T2(0), T2(0), T2(0)};
    for (std::size_t r = 1; r < pages.size(); ++r) {
      d = cross3(pages[0].wedge, pages[r].wedge);
      if (!is_zero(d))
        break;
    }
    if (is_zero(d))
      return ring;
    const auto pa = endpoint(reference.point_tag_0, reference.point_0);
    const auto pb = endpoint(reference.point_tag_1, reference.point_1);
    const nvec delta{T2(T1(pb[0]) - pa[0]), T2(T1(pb[1]) - pa[1]),
                     T2(T1(pb[2]) - pa[2])};
    if (tf::exact::dot_sign(d, delta) < 0)
      for (int c = 0; c < 3; ++c)
        d[std::size_t(c)] = -d[std::size_t(c)];
    int axis = 0;
    for (int c = 1; c < 3; ++c)
      if ((d[std::size_t(c)] < 0 ? -d[std::size_t(c)] : d[std::size_t(c)]) >
          (d[std::size_t(axis)] < 0 ? -d[std::size_t(axis)]
                                    : d[std::size_t(axis)]))
        axis = c;
    ring.axis = axis;
    ring.sign = sign_t2(d[std::size_t(axis)]);
    ring.valid = true;
    return ring;
  };

  auto task = [&](const auto &chunk, local_t &local) {
    for (const Index piece : chunk) {
      if (fences.fan[std::size_t(piece)] == char(0))
        continue;
      local.pieces.push_back(piece);

      // the piece's canonical pair, in the stream's flat language
      const auto &reference =
          arrangement.piece_definitions(immutable, piece)[0];
      const auto va =
          arrangement.flat_of(reference.point_tag_0, reference.point_0);

      local.occurrences.clear();
      for (const auto row : incidence.rows_of_piece[std::size_t(piece)]) {
        const auto t = row / Index(3);
        if (coplanar_of[std::size_t(t)] != Index(-1))
          continue;
        const auto s = std::size_t(row % Index(3));
        const auto face = face_of[std::size_t(t)];
        const auto dir = char(triangles[std::size_t(t)][s] == va);
        local.occurrences.push_back(
            {tf::arrangement::plane_arrangement_face_plane(arrangement,
                                                           immutable, face),
             face, row, dir, side_of(face, dir)});
      }
      // a carrier stands on both sides of the piece, so the page is
      // (plane, side) and the sort that groups it is the only one
      std::sort(local.occurrences.begin(), local.occurrences.end(),
                [](const occurrence_t &a, const occurrence_t &b) {
                  return std::tie(a.plane, a.side, a.face, a.row) <
                         std::tie(b.plane, b.side, b.face, b.row);
                });

      local.pages.clear();
      for (std::size_t begin = 0; begin < local.occurrences.size();) {
        std::size_t end = begin + 1;
        while (end < local.occurrences.size() &&
               local.occurrences[end].plane == local.occurrences[begin].plane &&
               local.occurrences[end].side == local.occurrences[begin].side)
          ++end;
        local.pages.push_back({wedge_of(local.occurrences[begin].plane,
                                        local.occurrences[begin].side),
                               local.occurrences[begin].plane, Index(begin),
                               Index(end - begin),
                               local.occurrences[begin].side});
        begin = end;
      }

      // two pages pair the same way in either order, so the ring has
      // nothing to decide and the authority is never read
      const std::size_t K = local.pages.size();
      if (K >= 3) {
        auto ring =
            tf::intersect::graph::make_plane_edge_radial_authority<Index, Int>(
                arrangement.piece_definitions(immutable, piece),
                descriptor_of_face, apply_to_face, get_mesh_point);
        if (!ring.valid) {
          ++local.refusals;
          ring = pages_ring(local.pages, reference);
        }
        if (!ring.valid) {
          // coplanar pack: two antipodal wedge classes vs page 0
          const nvec ref = local.pages[0].wedge;
          std::sort(local.pages.begin(), local.pages.end(),
                    [&](const page_t &a, const page_t &b) {
                      const int ca = same_sense(ref, a.wedge);
                      const int cb = same_sense(ref, b.wedge);
                      if (ca != cb)
                        return ca > cb;
                      return std::tie(a.plane, a.side) <
                             std::tie(b.plane, b.side);
                    });
        } else {
          const auto k0 = std::size_t((ring.axis + 1) % 3);
          const auto k1 = std::size_t((ring.axis + 2) % 3);
          const int d_sign = ring.sign;
          // the turn is ONE component of the wedges' cross, which is degree
          // four and has no rung on the ladder — but that component IS a
          // determinant, so its sign is read without the product being formed
          const auto ccw = [&](const nvec &a, const nvec &b) -> int {
            return tf::exact::det2_sign<Int>(a[k0], b[k1], a[k1], b[k0]) *
                   d_sign;
          };
          const nvec ref = local.pages[0].wedge;
          const auto angle_class = [&](const nvec &w) -> int {
            const int c = ccw(ref, w);
            if (c > 0)
              return 1;
            if (c < 0)
              return 3;
            return same_sense(ref, w) > 0 ? 0 : 2;
          };
          std::sort(local.pages.begin(), local.pages.end(),
                    [&](const page_t &a, const page_t &b) {
                      const int ka = angle_class(a.wedge);
                      const int kb = angle_class(b.wedge);
                      if (ka != kb)
                        return ka < kb;
                      if (ka == 1 || ka == 3) {
                        const int c = ccw(a.wedge, b.wedge);
                        if (c != 0)
                          return c > 0;
                      }
                      return std::tie(a.plane, a.side) <
                             std::tie(b.plane, b.side);
                    });
        }
      }

      local.page_counts.push_back(Index(K));
      for (const auto &page : local.pages) {
        local.row_counts.push_back(page.count);
        for (auto at = page.begin; at < page.begin + page.count; ++at) {
          local.rows.push_back(local.occurrences[std::size_t(at)].row);
          local.dirs.push_back(local.occurrences[std::size_t(at)].dir);
        }
      }
    }
  };

  auto aggregate = [](const local_t &local, plane_radial_fans<Index> &result) {
    tf::core::append(local.pieces, result.pieces);
    for (const auto count : local.page_counts)
      result.page_offsets.push_back(
          result.page_offsets[result.page_offsets.size() - 1] + count);
    auto &offsets = result.rows.offsets_buffer();
    for (const auto count : local.row_counts)
      offsets.push_back(offsets[offsets.size() - 1] + count);
    tf::core::append(local.rows, result.rows.data_buffer());
    tf::core::append(local.dirs, result.dirs);
    result.n_refused += local.refusals;
  };

  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(Index(fences.fan.size())), out, local_t{}, task,
      aggregate);
  return out;
}

} // namespace tf::csg::graph
