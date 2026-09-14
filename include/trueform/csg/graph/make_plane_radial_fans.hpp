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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/det2_sign.hpp"
#include "../../exact/dot_sign.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_support.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_edge_radial_authority.hpp"
#include "../../topology/directed_edge_id_in_face.hpp"
#include "../../topology/manifold_edge_peer.hpp"
#include "./make_plane_triangle_faces.hpp"
#include "./triangle_component_labels.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace tf::csg::graph {

/// The radial fans of the arrangement: one fan per ring around a
/// non-manifold or seam edge, its pages in radial order around the edge.
///
/// A PAGE is one carrier on one of its two sides — the half-plane the
/// sectors meet at — and it carries every live occurrence sitting on
/// it, so triangulation multiplicity and stack depth cost the ring
/// nothing. A ring holds BOTH STATES of a sheet: a cut sheet's pages
/// come from the plane world, and an uncut sheet presents itself as a
/// carrier of its own — its exact supported normal, in its own winding
/// — so one radial order sorts them together. `pieces[k]` is the k-th
/// fan's piece ticket, `-1` for a ring on a source edge no piece names
/// (every sheet uncut); `[page_offsets[k], page_offsets[k + 1])` are its
/// pages, radially sorted. `rows[p]` is page `p`'s occurrence rows: a
/// cut occurrence is `triangle * 3 + slot`, an uncut sheet is
/// `-(component + 1)` — its surface component, stated directly, since
/// no stream row names it. `dirs` walks in lockstep with the row data —
/// 1 when the occurrence traverses the edge in the ring's canonical
/// order. Every fan piece has a fan, whatever its size: a two-page fan
/// closes its curves and divides like any other.
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
///
/// THE UNCUT SHEETS. A non-manifold boundary definition names its source
/// directed side `(tag_other, object_other, side)`; the side's edge, read
/// through the form's own faces, names every incident face, and a face the
/// surface labels still own (`polygon_labels != none`) is a sheet the cut
/// world never saw. It joins the ring as `(supported normal, +1, traversal
/// against the piece's key order)` — the same triple a carrier states — and
/// a face whose support is a line states no half-plane and is skipped, as a
/// collapsed face is in the cut world. A source edge NO piece names — its
/// representative marked in the form's `manifold_edge_link`, every incident
/// face uncut — is a ring carrier of its own, ordered about the edge's own
/// lattice line.
template <typename Index, typename Int, typename Immutable, typename Labels,
          typename GetMeshPoint, typename ApplyToFace, typename ApplyToForm>
auto make_plane_radial_fans(
    const tf::arrangement::plane_arrangement<Index, Int> &arrangement,
    const Immutable &immutable,
    const tf::arrangement::plane_piece_incidence<Index> &incidence,
    const tf::arrangement::plane_piece_fences &fences, const Labels &labels,
    const GetMeshPoint &get_mesh_point, const ApplyToFace &apply_to_face,
    const ApplyToForm &apply_to_form, Index n_tags)
    -> plane_radial_fans<Index> {
  constexpr bool with_uncut = !std::is_same_v<Labels, tf::none_t>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using nvec = std::array<T2, 3>;

  plane_radial_fans<Index> out;
  out.page_offsets.push_back(Index(0));
  out.rows.offsets_buffer().push_back(Index(0));
  const bool any_fan = std::any_of(fences.fan.begin(), fences.fan.end(),
                                   [](char fan) { return fan != char(0); });

  const auto coplanar_of = arrangement.coplanar_of();
  const auto triangles = arrangement.triangles();
  const auto face_of = make_plane_triangle_faces(arrangement);

  const auto get_base_point = [&](std::int16_t tag, Index id) {
    return tag < std::int16_t(0) ? immutable.point_of(id, get_mesh_point)
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

  // The carrier answers every geometric question about its members: a
  // member's stored winding is the sign of its normal against its
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
  // The sense of two wedges a turn has proven parallel: one is a multiple
  // of the other, so the sign of one shared component's product is the
  // sign of their dot — which a degree-four dot cannot be formed to
  // answer.
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

  // the turn is one component of the wedges' cross — degree four, no
  // rung on the ladder — but that component is a determinant, so its
  // sign is read without the product being formed
  const auto radial_sort = [&](auto &pages, int axis, int d_sign) {
    const auto k0 = std::size_t((axis + 1) % 3);
    const auto k1 = std::size_t((axis + 2) % 3);
    const auto ccw = [&](const nvec &a, const nvec &b) -> int {
      return tf::exact::det2_sign<Int>(a[k0], b[k1], a[k1], b[k0]) * d_sign;
    };
    const nvec ref = pages[0].wedge;
    const auto angle_class = [&](const nvec &w) -> int {
      const int c = ccw(ref, w);
      if (c > 0)
        return 1;
      if (c < 0)
        return 3;
      return same_sense(ref, w) > 0 ? 0 : 2;
    };
    std::sort(pages.begin(), pages.end(), [&](const auto &a, const auto &b) {
      const int ka = angle_class(a.wedge);
      const int kb = angle_class(b.wedge);
      if (ka != kb)
        return ka < kb;
      if (ka == 1 || ka == 3) {
        const int c = ccw(a.wedge, b.wedge);
        if (c != 0)
          return c > 0;
      }
      return std::tie(a.plane, a.side) < std::tie(b.plane, b.side);
    });
  };

  /// One live occurrence: the row that states it, the face it came out
  /// of, and the page it sits on — its carrier plane and the side of it.
  /// An uncut sheet is its own carrier: its plane is a ring-local ticket
  /// `-(ordinal + 1)` into the ring's normal table, and its row states
  /// the surface component directly, `-(component + 1)`.
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
    tf::buffer<nvec> uncut_normals;
    tf::buffer<std::array<Index, 3>> nm_sides;
    tf::buffer<std::array<Index, 2>> uncut_seen;
    Index refusals = 0;
  };

  const auto negate = [](const nvec &v) -> nvec {
    return {T2(-v[0]), T2(-v[1]), T2(-v[2])};
  };

  // The ill-posed piece's ring: several carrier lines welded into one
  // canonical identity leave the piece with no line of its own, so the turn
  // is taken from the pages themselves: the cross of the first two
  // independent wedges, turned to the order the resolved endpoints stand in.
  //
  // The cross of two wedges is degree four, which has no rung on the
  // ladder, and the dot against it reads its operands past the width its
  // own contract states: on a carrier standing at the lattice's full span
  // both leave their type. A ring read off the piece's own endpoints is
  // exact and needs neither, and it is not what this states — it moves
  // corpus pairs in both directions, so it is an open question, not a fix.
  const auto pages_ring = [&](const tf::buffer<page_t> &pages,
                              const auto &reference)
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
      const auto definitions = arrangement.piece_definitions(immutable, piece);
      const auto &reference = definitions[0];
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

      // A non-manifold boundary definition names its
      // source side; the side's own corners, turned by the reversed
      // flag, state the piece's key order as a source vertex pair —
      // split-invariant, since a sub-piece re-flips the bit with its
      // key — so an uncut incident face's traversal bit is stated in
      // the same language as a cut occurrence's.
      constexpr bool ring_uncut = !std::is_same_v<Labels, tf::none_t>;
      if constexpr (ring_uncut) {
        local.uncut_normals.clear();
        local.nm_sides.clear();
        local.uncut_seen.clear();
        for (const auto &def : definitions) {
          if (def.side < std::int16_t(0) ||
              (def.flags &
               tf::intersect::graph::plane_edge_non_manifold_flag) == 0)
            continue;
          Index ka = Index(-1);
          Index kb = Index(-1);
          apply_to_face(int(def.tag_other), def.object_other,
                        [&](const auto &corners) {
                          const auto n = corners.size();
                          if (std::size_t(def.side) >= n)
                            return;
                          ka = Index(corners[std::size_t(def.side)]);
                          kb = Index(corners[(std::size_t(def.side) + 1) % n]);
                        });
          if (ka == Index(-1) || ka == kb)
            continue;
          if ((def.flags & tf::intersect::graph::plane_edge_reversed_flag) != 0)
            std::swap(ka, kb);
          const std::array<Index, 3> stated{
              Index(def.tag_other), ka < kb ? ka : kb, ka < kb ? kb : ka};
          bool named = false;
          for (const auto &seen : local.nm_sides)
            named = named || seen == stated;
          if (named)
            continue;
          local.nm_sides.push_back(stated);
          const auto tag = Index(def.tag_other);
          auto poly_labels = labels.polygon_labels(tag);
          apply_to_form(tag, [&](const auto &form) {
            const auto faces_t = form.faces();
            auto &&fm = form.face_membership();
            for (const auto f2 : fm[ka]) {
              const auto face2 = faces_t[f2];
              const auto sz = Index(face2.size());
              const bool along =
                  tf::directed_edge_id_in_face(ka, kb, face2) != sz;
              if (!along && tf::directed_edge_id_in_face(kb, ka, face2) == sz)
                continue;
              if (poly_labels[std::size_t(f2)] ==
                  triangle_component_labels<Index>::none_label)
                continue;
              const std::array<Index, 2> sheet{tag, Index(f2)};
              bool present = false;
              for (const auto &seen : local.uncut_seen)
                present = present || seen == sheet;
              if (present)
                continue;
              local.uncut_seen.push_back(sheet);
              tf::exact::plane_support<Int> support;
              for (std::size_t c = 0; c < std::size_t(face2.size()); ++c) {
                support.offer(get_mesh_point(int(tag), Index(face2[c])));
                if (support.size == 3)
                  break;
              }
              if (support.size != 3)
                continue; // a line states no half-plane
              const auto ordinal = Index(local.uncut_normals.size());
              local.uncut_normals.push_back(support.normal);
              local.occurrences.push_back(
                  {Index(-(ordinal + 1)), Index(f2),
                   Index(-(poly_labels[std::size_t(f2)] + 1)), char(along),
                   static_cast<signed char>(along ? 1 : -1)});
            }
          });
        }
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
        const occurrence_t &first = local.occurrences[begin];
        nvec wedge;
        if ((first.plane) < Index(0)) {
          wedge = local.uncut_normals[std::size_t(-first.plane) - 1];
          if (first.side < 0)
            wedge = negate(wedge);
        } else {
          wedge = wedge_of(first.plane, first.side);
        }
        local.pages.push_back(
            {wedge, first.plane, Index(begin), Index(end - begin), first.side});
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
          radial_sort(local.pages, ring.axis, ring.sign);
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

  if (any_fan)
    tf::blocked_reduce_sequenced_aggregate(
        tf::make_sequence_range(Index(fences.fan.size())), out, local_t{}, task,
        aggregate);

  // The edge-grain carriers: source non-manifold sides no piece names.
  // The link already marks one representative per non-manifold edge, so
  // the discovery is one comparison per side of every face; a sheet of
  // the edge in the cut world means its defs put every ring of the edge
  // under the piece tier, so only the all-uncut edge rings here.
  if constexpr (with_uncut) {
    struct uncut_sheet_t {
      nvec normal;
      Index label;
      char dir;
    };
    tf::buffer<std::array<Index, 2>> representatives;
    tf::buffer<uncut_sheet_t> sheets;
    tf::buffer<page_t> pages;
    for (Index tag = 0; tag < n_tags; ++tag) {
      auto poly_labels = labels.polygon_labels(tag);
      apply_to_form(tag, [&](const auto &form) {
        const auto faces_t = form.faces();
        auto &&mel = form.manifold_edge_link();
        auto &&fm = form.face_membership();
        representatives.clear();
        tf::generic_generate(
            tf::make_sequence_range(Index(faces_t.size())), representatives,
            [&](Index f, tf::buffer<std::array<Index, 2>> &found) {
              const auto peers = mel[f];
              for (Index s = 0; s < Index(peers.size()); ++s)
                if (peers[std::size_t(s)].face_peer ==
                    tf::manifold_edge_peer<Index>::non_manifold_representative)
                  found.push_back({f, s});
            },
            tf::checked);
        for (const auto &rep : representatives) {
          const auto face = faces_t[rep[0]];
          const auto n = Index(face.size());
          const auto i = Index(face[std::size_t(rep[1])]);
          const auto j = Index(face[std::size_t((rep[1] + 1) % n)]);
          if (i == j)
            continue;
          sheets.clear();
          bool any_cut = false;
          for (const auto f2 : fm[i]) {
            const auto face2 = faces_t[f2];
            const auto sz = Index(face2.size());
            const bool along = tf::directed_edge_id_in_face(i, j, face2) != sz;
            if (!along && tf::directed_edge_id_in_face(j, i, face2) == sz)
              continue;
            if (poly_labels[std::size_t(f2)] ==
                triangle_component_labels<Index>::none_label) {
              any_cut = true;
              break;
            }
            tf::exact::plane_support<Int> support;
            for (std::size_t c = 0; c < std::size_t(face2.size()); ++c) {
              support.offer(get_mesh_point(int(tag), Index(face2[c])));
              if (support.size == 3)
                break;
            }
            if (support.size != 3)
              continue; // a line states no half-plane
            sheets.push_back({support.normal,
                              Index(poly_labels[std::size_t(f2)]),
                              char(along)});
          }
          if (any_cut)
            continue;
          const auto pi = get_mesh_point(int(tag), i);
          const auto pj = get_mesh_point(int(tag), j);
          const std::array<T1, 3> line{T1(pj[0]) - pi[0], T1(pj[1]) - pi[1],
                                       T1(pj[2]) - pi[2]};
          const auto magnitude = [](const T1 &value) {
            return value < T1(0) ? T1(-value) : value;
          };
          int axis = 0;
          for (int c = 1; c < 3; ++c)
            if (magnitude(line[std::size_t(c)]) >
                magnitude(line[std::size_t(axis)]))
              axis = c;
          if (line[std::size_t(axis)] == T1(0))
            continue;
          const int d_sign = line[std::size_t(axis)] > T1(0) ? 1 : -1;
          pages.clear();
          for (std::size_t k = 0; k < sheets.size(); ++k) {
            const auto &sheet = sheets[k];
            pages.push_back({sheet.dir ? sheet.normal : negate(sheet.normal),
                             Index(-(Index(k) + 1)), Index(k), Index(1),
                             static_cast<signed char>(sheet.dir ? 1 : -1)});
          }
          if (pages.size() >= 3)
            radial_sort(pages, axis, d_sign);
          out.pieces.push_back(Index(-1));
          out.page_offsets.push_back(
              out.page_offsets[out.page_offsets.size() - 1] +
              Index(pages.size()));
          auto &offsets = out.rows.offsets_buffer();
          for (const auto &page : pages) {
            offsets.push_back(offsets[offsets.size() - 1] + Index(1));
            out.rows.data_buffer().push_back(
                Index(-(sheets[std::size_t(page.begin)].label + 1)));
            out.dirs.push_back(sheets[std::size_t(page.begin)].dir);
          }
        }
      });
    }
  }
  return out;
}

/// The cut tier's fans alone — no surface labels in hand, so no uncut
/// sheets and no edge-grain carriers: the arrangement's own read.
template <typename Index, typename Int, typename Immutable,
          typename GetMeshPoint, typename ApplyToFace>
auto make_plane_radial_fans(
    const tf::arrangement::plane_arrangement<Index, Int> &arrangement,
    const Immutable &immutable,
    const tf::arrangement::plane_piece_incidence<Index> &incidence,
    const tf::arrangement::plane_piece_fences &fences,
    const GetMeshPoint &get_mesh_point, const ApplyToFace &apply_to_face)
    -> plane_radial_fans<Index> {
  return make_plane_radial_fans<Index, Int>(arrangement, immutable, incidence,
                                            fences, tf::none, get_mesh_point,
                                            apply_to_face, tf::none, Index(0));
}

} // namespace tf::csg::graph
