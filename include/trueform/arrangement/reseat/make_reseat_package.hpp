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
#include "../../core/buffer.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/elect_identity_components.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/name_plane_carriers.hpp"
#include "../../intersect/graph/near_identity_rows.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/seat_named_planes.hpp"
#include "../identity/identity_substrate.hpp"
#include "./make_reseat_delta_forms.hpp"
#include "./make_reseat_delta_world.hpp"
#include "./reseat_pieces.hpp"
#include "./reseat_refusal.hpp"
#include "./state_reseat_reach.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement::reseat {

/// The carve as statements to the standing world, in its own names.
///
/// The carve states only what it creates. A corner the delta holds speaks
/// the world's own name for it and is never elected again — its merged
/// name, or the anchor its identified class is named by where a standing
/// row names it, as the cut world's loops do — and the delta's anchor of a
/// corner is that corner. A seat or a class the delta created speaks the
/// created identity standing at its position, so a coincidence the world
/// already holds stays the gate's. Only a seat or a class no created
/// identity stands at mints, and the originals there retire into the mint.
///
/// - `minted`: identities the carve creates, `(-1, minted_base + k)` in
///   every row below;
/// - `welds`: `{kind, source, target}` — an identity the carve moved is
///   merged into the one standing where the carve put it;
/// - `retired`: world faces the carve replaced by pieces — a source face
///   standing as pieces retires all of them;
/// - `restated`: world faces whose rows the carve restates whole;
/// - the pieces: one promoted face each, `(tag, n_faces(tag) + k)` past
///   every piece the world already names, seated on `piece_plane_of` — a
///   standing plane or a new one past `n_world_planes`, framed by
///   `new_frames` and named by `new_plane_names`. A piece is seated against
///   every plane the substrate names, so a name only the plane tier holds
///   is refused (`reseat_refusal::plane_promoted`) rather than founded twice;
/// - `rows`: every row of every piece and restated face; a piece's rows
///   name its face `n_world_faces + k`;
/// - `near_rows`: when the world asks, the near table between the mints and
///   the standing identities the carve can meet, as one deferred block whose
///   floor is the first mint (@ref
///   tf::intersect::graph::near_identity_rows).
template <typename Index, typename Int> struct reseat_package {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using pt3_t = tf::exact::pt3<Int>;

  bool ok = false;
  Index minted_base = 0;
  tf::buffer<pt3_t> minted;
  tf::buffer<std::array<Index, 3>> welds;
  tf::buffer<Index> retired;
  tf::buffer<Index> restated;
  tf::buffer<tf::intersect::graph::face_descriptor<Index>> piece_descriptors;
  tf::buffer<std::array<Index, 2>> piece_sources;
  tf::buffer<std::array<std::int8_t, 3>> piece_sides;
  tf::buffer<std::array<piece_corner<Index>, 3>> piece_corners;
  tf::buffer<std::int8_t> piece_orientations;
  tf::buffer<Index> piece_plane_of;
  tf::buffer<tf::exact::plane_frame<Int>> new_frames;
  tf::buffer<tf::exact::canonical_plane<Int>> new_plane_names;
  tf::buffer<def_t> rows;
  /// the standing planes the carve's rows can reach, ascending
  tf::buffer<Index> planes;
  /// faces whose sides the delta split differently from the world: the
  /// carve restates them too (`reseat_refusal::reach_unbounded`)
  tf::buffer<std::array<Index, 2>> promote;
  tf::intersect::graph::near_identity_rows<Index, pt3_t> near_rows;
  reseat_refusal refusal = reseat_refusal::none;
};

template <typename Index, typename Int, typename World, typename MeshReader,
          typename ApplyToForm>
auto make_reseat_package(
    const World &world,
    const tf::arrangement::identity_substrate<Index, Int> &world_identities,
    const standing_pieces<Index> &standing,
    const reseat_pieces<Index, Int> &pieces, const reseat_reach<Index> &reach,
    const reseat_delta_forms<Index, Int> &forms,
    const reseat_delta_world<Index, Int> &delta, const MeshReader &reader,
    const ApplyToForm &apply_to_form, Index n_tags)
    -> reseat_package<Index, Int> {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using pt3_t = tf::exact::pt3<Int>;
  using ref_t = std::array<Index, 2>; // {tag (-1 created), id}
  using key_t = std::array<Index, 2>;
  reseat_package<Index, Int> out;
  out.minted_base = world.n_created_points(world_identities);
  const auto &dw = delta.world;
  const auto &dw_identities = delta.identities;
  // the carve names a world identity as the graph stands, `P∘L`, in the
  // currency of the local tier whose writer it is; a name foreign to it is
  // read as the key it was asked of, and refuses the carve
  bool foreign = false;
  const auto world_name = [&world_identities, &foreign,
                           n_created = out.minted_base](Index source,
                                                        Index key) {
    const auto target = world_identities.composed_target(source, key);
    if (target < n_created)
      return target;
    foreign = true;
    return Index(-1);
  };
  const auto dw_merges = tf::make_range(dw_identities.local_merges());
  const auto dreader = delta.reader();
  const auto dforms = delta.apply_to_form();

  const world_face_lookup<Index, World> world_face_of(world);
  for (const auto &c : pieces.carved) {
    const auto range = standing.of_source(c[0], c[1]);
    for (auto p = range[0]; p < range[1]; ++p)
      out.retired.push_back(standing.faces[p]);
    if (range[0] != range[1])
      continue;
    const auto face = world_face_of(c[0], c[1]);
    if (face != Index(-1))
      out.retired.push_back(face);
  }
  // a reached face the world never named enters it, as every face a carve
  // reaches must
  tf::buffer<std::array<Index, 2>> entered;
  for (const auto &r : reach.restated) {
    const auto face = world_face_of(r[0], r[1]);
    if (face == Index(-1)) {
      entered.push_back(r);
      continue;
    }
    out.restated.push_back(face);
  }
  const auto entered_rank = [&](Index tag, Index object) -> Index {
    const auto at = std::lower_bound(entered.begin(), entered.end(),
                                     std::array<Index, 2>{tag, object});
    if (at == entered.end() || (*at)[0] != tag || (*at)[1] != object)
      return Index(-1);
    return Index(at - entered.begin());
  };
  std::sort(out.retired.begin(), out.retired.end());
  std::sort(out.restated.begin(), out.restated.end());

  // the pieces' promoted names, past each tag's source faces and the pieces
  // the world already names
  tf::buffer<Index> n_source_faces;
  for (Index t = 0; t < n_tags; ++t)
    apply_to_form(t, [&](const auto &form) {
      n_source_faces.push_back(Index(form.faces().size()));
    });
  const auto n_world_faces = world.n_faces();
  tf::buffer<Index> piece_rank;
  piece_rank.allocate(pieces.pieces.size());
  {
    tf::buffer<Index> next = n_source_faces;
    for (Index t = 0; t < n_tags; ++t)
      next[std::size_t(t)] += standing.stated_of(t);
    for (std::size_t k = 0; k < pieces.pieces.size(); ++k) {
      const auto &piece = pieces.pieces[k];
      piece_rank[k] = next[std::size_t(piece.tag)]++;
      out.piece_descriptors.push_back({piece.tag, piece_rank[k]});
      out.piece_sources.push_back({piece.tag, piece.source});
      out.piece_sides.push_back(piece.sides);
    }
    for (const auto &e : entered)
      out.piece_descriptors.push_back({e[0], e[1]});
  }
  const auto n_pieces = Index(pieces.pieces.size());

  // the delta's faces by name, and each one's world meaning
  const auto origin_of =
      [&](Index dtag, Index dobject) -> const delta_face_origin<Index> & {
    return forms.faces[std::size_t(dtag)][std::size_t(dobject)];
  };

  // the standing identities the delta can meet, by lattice point: every
  // endpoint of every row of every plane a delta face stands on
  tf::buffer<Index> planes;
  for (const auto face : out.retired)
    planes.push_back(world.plane_of_face(face));
  for (const auto face : out.restated)
    planes.push_back(world.plane_of_face(face));
  for (const auto &key : reach.neighbours) {
    const auto face = world_face_of(key[0], key[1]);
    if (face != Index(-1))
      planes.push_back(world.plane_of_face(face));
  }
  std::sort(planes.begin(), planes.end());
  planes.erase_till_end(std::unique(planes.begin(), planes.end()));
  out.planes = planes;
  const auto world_defs = world.edge_defs();
  const auto vertex_offsets = world.vertex_offsets();
  const auto n_flat = vertex_offsets[vertex_offsets.size() - 1];
  const auto flat_of = [&](const ref_t &r) {
    return r[0] < 0 ? n_flat + r[1] : vertex_offsets[std::size_t(r[0])] + r[1];
  };
  const auto point_of_ref = [&](const ref_t &r) -> pt3_t {
    if (r[0] >= 0)
      return reader(int(r[0]), r[1]);
    if (r[0] == Index(seat_corner_tag))
      return pieces.seats[std::size_t(r[1])];
    if (r[1] >= out.minted_base)
      return out.minted[std::size_t(r[1] - out.minted_base)];
    return world.point_of(world_identities, r[1], reader);
  };
  struct standing_t {
    pt3_t point;
    ref_t ref;
  };
  tf::buffer<standing_t> identities;
  for (const auto plane : planes)
    for (const auto row : world.plane_edges(plane)) {
      const auto &def = world_defs[std::size_t(row)];
      const ref_t a{Index(def.point_tag_0), def.point_0};
      const ref_t b{Index(def.point_tag_1), def.point_1};
      identities.push_back({point_of_ref(a), a});
      identities.push_back({point_of_ref(b), b});
    }
  for (const auto &vertices : forms.vertices)
    for (const auto &c : vertices) {
      if (c.tag == seat_corner_tag)
        continue;
      std::int16_t merged_tag;
      Index merged_id;
      tf::intersect::graph::plane_targeted_endpoint(
          world_name, world.vertex_offsets(), c.tag, c.id, merged_tag,
          merged_id);
      const ref_t r{Index(merged_tag), merged_id};
      identities.push_back({point_of_ref(r), r});
    }
  const auto point_less = [](const pt3_t &a, const pt3_t &b) {
    return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
  };
  std::sort(identities.begin(), identities.end(),
            [&](const standing_t &x, const standing_t &y) {
              if (point_less(x.point, y.point))
                return true;
              if (point_less(y.point, x.point))
                return false;
              return x.ref < y.ref;
            });
  identities.erase_till_end(std::unique(
      identities.begin(), identities.end(),
      [](const standing_t &x, const standing_t &y) { return x.ref == y.ref; }));

  tf::buffer<ref_t> resolved_of;
  const auto dw_created = dw.n_created_points(dw_identities);
  resolved_of.allocate(std::size_t(dw_created));
  tf::buffer<char> resolved;
  resolved.allocate_and_initialize(std::size_t(dw_created), char(0));
  // the standing identities at one lattice point, the created ones first
  const auto standing_at = [&](const pt3_t &point) {
    const auto first =
        std::lower_bound(identities.begin(), identities.end(), point,
                         [&](const standing_t &x, const pt3_t &p) {
                           return point_less(x.point, p);
                         });
    auto last = first;
    while (last != identities.end() && last->point == point)
      ++last;
    return tf::make_range(first, last);
  };
  // the rows the elections state, in the weld currency
  tf::buffer<std::array<Index, 3>> retirements;
  // every point the carve can mint is a seat or a delta class, so the
  // distinct positions are stated once and a mint is found by search; a
  // position mints the first time it is asked, which keeps mint order
  tf::buffer<pt3_t> mint_positions;
  mint_positions.reserve(pieces.seats.size() + std::size_t(dw_created));
  for (const auto &seat : pieces.seats)
    mint_positions.push_back(seat);
  for (Index id = 0; id < dw_created; ++id)
    mint_positions.push_back(dw.point_of(dw_identities, id, dreader));
  std::sort(mint_positions.begin(), mint_positions.end(), point_less);
  mint_positions.erase_till_end(
      std::unique(mint_positions.begin(), mint_positions.end()));
  tf::buffer<Index> mint_of_position;
  mint_of_position.allocate_and_initialize(mint_positions.size(), Index(-1));
  const auto position_of = [&](const pt3_t &point) {
    return std::size_t(std::lower_bound(mint_positions.begin(),
                                        mint_positions.end(), point,
                                        point_less) -
                       mint_positions.begin());
  };
  const auto mint_at = [&](const pt3_t &point) -> ref_t {
    auto &k = mint_of_position[position_of(point)];
    if (k == Index(-1)) {
      k = Index(out.minted.size());
      out.minted.push_back(point);
    }
    return {Index(-1), out.minted_base + k};
  };
  // the election at a point the carve creates
  const auto key_of = [&flat_of](const standing_t &x) {
    return x.ref[0] < 0
               ? tf::intersect::graph::standing_identity_key<Index>(x.ref[1])
               : tf::intersect::graph::original_identity_key<Index>(
                     flat_of(x.ref));
  };
  const auto elect = [&](const auto &run, const pt3_t &site) -> ref_t {
    if (run[0].ref[0] < 0)
      return run[0].ref;
    const auto mint =
        [&mint_at, &site](const tf::intersect::graph::identity_key<Index> &) {
          return mint_at(site)[1];
        };
    return {Index(-1),
            tf::intersect::graph::elect_identity_component<Index>(
                tf::make_mapped_range(run, key_of), mint, retirements)};
  };
  const auto merged_of = [&](std::int16_t tag, Index id) -> ref_t {
    std::int16_t merged_tag;
    Index merged_id;
    tf::intersect::graph::plane_targeted_endpoint(
        world_name, vertex_offsets, tag, id, merged_tag, merged_id);
    return {Index(merged_tag), merged_id};
  };
  const auto corner_name = [&](const piece_corner<Index> &c) -> ref_t {
    if (c.tag == seat_corner_tag) {
      const auto &point = pieces.seats[std::size_t(c.id)];
      const auto run = standing_at(point);
      return run.size() == 0 ? mint_at(point) : elect(run, point);
    }
    const auto merged = merged_of(c.tag, c.id);
    if (merged[0] < 0)
      return merged;
    const auto canonical = world.intersections().canonical_vertex(
        std::int16_t(merged[0]), merged[1]);
    for (const auto &x : standing_at(point_of_ref(merged)))
      if (x.ref[0] < 0 && x.ref[1] < world.n_vertex_points()) {
        const auto &anchor = world.vertex_anchor(x.ref[1]);
        if (vertex_offsets[std::size_t(anchor.tag)] + anchor.vid == canonical)
          return merged_of(std::int16_t(-1), x.ref[1]);
      }
    return merged;
  };
  const auto resolve = [&](std::int16_t tag, Index id) -> ref_t {
    if (tag >= 0)
      return corner_name(forms.vertices[std::size_t(tag)][std::size_t(id)]);
    if (resolved[std::size_t(id)])
      return resolved_of[std::size_t(id)];
    ref_t r;
    if (id < dw.n_vertex_points()) {
      const auto &anchor = dw.vertex_anchor(id);
      r = corner_name(
          forms.vertices[std::size_t(anchor.tag)][std::size_t(anchor.vid)]);
    } else {
      const auto point = dw.point_of(dw_identities, id, dreader);
      const auto run = standing_at(point);
      r = run.size() == 0 ? mint_at(point) : elect(run, point);
    }
    resolved[std::size_t(id)] = char(1);
    resolved_of[std::size_t(id)] = r;
    return r;
  };

  const auto other_of = [&](Index dtag, Index dobject) -> key_t {
    const auto &o = origin_of(dtag, dobject);
    if (o.role == delta_role::piece)
      return {o.tag, piece_rank[std::size_t(o.id)]};
    return {o.tag, o.id};
  };
  // the delta's rows, in the world's names
  const auto delta_defs = dw.edge_defs();
  for (const auto &ddef : delta_defs) {
    const auto &descriptor = dw.descriptor(ddef.face);
    const auto &o = origin_of(Index(descriptor.tag), descriptor.object);
    if (o.role == delta_role::neighbour)
      continue;
    auto a = resolve(ddef.point_tag_0, ddef.point_0);
    auto b = resolve(ddef.point_tag_1, ddef.point_1);
    if (a == b)
      continue;
    def_t def = ddef;
    if (b < a) {
      std::swap(a, b);
      def.flags ^= tf::intersect::graph::plane_edge_reversed_flag;
      if (def.flags & tf::intersect::graph::plane_edge_radial_flag)
        def.flags ^= tf::intersect::graph::plane_edge_radial_reversed_flag;
    }
    def.point_tag_0 = std::int16_t(a[0]);
    def.point_0 = a[1];
    def.point_tag_1 = std::int16_t(b[0]);
    def.point_1 = b[1];
    if (o.role == delta_role::piece)
      def.face = n_world_faces + Index(o.id);
    else {
      def.face = world_face_of(o.tag, o.id);
      if (def.face == Index(-1))
        def.face = n_world_faces + n_pieces + entered_rank(o.tag, o.id);
    }
    const auto other = other_of(Index(ddef.tag_other), ddef.object_other);
    def.tag_other = std::int16_t(other[0]);
    def.object_other = other[1];
    def.id = Index(-1);
    out.rows.push_back(def);
  }

  // a neighbour a piece meets is a face the carve changed: it is restated
  {
    const auto blocks = dw.intersections().intersections();
    for (Index face = 0; face < dw.n_cut_faces(); ++face) {
      const auto &d = dw.descriptor(face);
      const auto &o = origin_of(Index(d.tag), d.object);
      if (o.role != delta_role::neighbour)
        continue;
      for (const auto &record : blocks[std::size_t(face)]) {
        const auto &other =
            origin_of(Index(record.tag_other), record.object_other);
        if (other.role == delta_role::piece) {
          out.promote.push_back({o.tag, o.id});
          break;
        }
      }
    }
    if (out.promote.size() != 0) {
      std::sort(out.promote.begin(), out.promote.end());
      out.promote.erase_till_end(
          std::unique(out.promote.begin(), out.promote.end()));
      out.refusal = reseat_refusal::reach_unbounded;
      return out;
    }
  }
  // a side the carve's rows split at a point the world
  // never named is split for every face holding that edge, so a face holding
  // it that keeps its world rows is reached across the side
  {
    tf::buffer<std::array<Index, 3>> split_sides; // tag, a, b (a < b)
    for (const auto &ddef : delta_defs) {
      if (ddef.side < 0)
        continue;
      const auto &descriptor = dw.descriptor(ddef.face);
      const auto &o = origin_of(Index(descriptor.tag), descriptor.object);
      if (o.role == delta_role::neighbour)
        continue;
      const auto a = resolve(ddef.point_tag_0, ddef.point_0);
      const auto b = resolve(ddef.point_tag_1, ddef.point_1);
      const bool fresh = (a[0] < 0 && a[1] >= out.minted_base) ||
                         (b[0] < 0 && b[1] >= out.minted_base);
      if (!fresh)
        continue;
      // the source edge the side lies on, by its source face's own corners:
      // a piece states the source side each of its sides lies on
      Index source = o.id;
      std::int8_t side = std::int8_t(ddef.side);
      if (o.role == delta_role::piece) {
        const auto &piece = pieces.pieces[std::size_t(o.id)];
        source = piece.source;
        side = piece.sides[std::size_t(ddef.side)];
      } else {
        const auto at = standing.of_key(o.tag, o.id);
        if (at != Index(-1)) {
          source = standing.pieces[std::size_t(at)].source;
          side = standing.pieces[std::size_t(at)].sides[std::size_t(ddef.side)];
        }
      }
      if (side < 0)
        continue;
      std::array<Index, 2> ends{};
      apply_to_form(o.tag, [&](const auto &form) {
        const auto f = form.faces()[source];
        ends = {Index(f[std::size_t(side)]),
                Index(f[(std::size_t(side) + 1) % 3])};
      });
      split_sides.push_back(
          {o.tag, std::min(ends[0], ends[1]), std::max(ends[0], ends[1])});
    }
    std::sort(split_sides.begin(), split_sides.end());
    split_sides.erase_till_end(
        std::unique(split_sides.begin(), split_sides.end()));
    // a source face standing as pieces is stated when every piece of it is
    const auto stated = [&](Index tag, Index face) {
      const auto in = [](const tf::buffer<key_t> &set, const key_t &key) {
        return std::binary_search(set.begin(), set.end(), key);
      };
      if (in(reach.restated, {tag, face}) || in(pieces.carved, {tag, face}))
        return true;
      const auto range = standing.of_source(tag, face);
      if (range[0] == range[1])
        return false;
      for (auto p = range[0]; p < range[1]; ++p)
        if (!in(reach.restated, {tag, standing.keys[p]}))
          return false;
      return true;
    };
    for (const auto &e : split_sides)
      apply_to_form(e[0], [&](const auto &form) {
        const auto faces = form.faces();
        for (const auto other : form.face_membership()[e[1]]) {
          const auto g = faces[other];
          bool holds = false;
          for (std::size_t j = 0; j < 3; ++j)
            holds = holds || Index(g[j]) == e[2];
          if (holds && !stated(e[0], Index(other)))
            out.promote.push_back({e[0], Index(other)});
        }
      });
    std::sort(out.promote.begin(), out.promote.end());
    out.promote.erase_till_end(
        std::unique(out.promote.begin(), out.promote.end()));
    if (out.promote.size() != 0) {
      out.refusal = reseat_refusal::reach_unbounded;
      return out;
    }
  }
  // the pieces' corners in the world's names
  for (const auto &piece : pieces.pieces) {
    std::array<piece_corner<Index>, 3> corners{};
    for (std::size_t c = 0; c < 3; ++c) {
      const auto &corner = piece.corners[c];
      ref_t r{Index(corner.tag), corner.id};
      if (corner.tag == seat_corner_tag) {
        const auto &seat = pieces.seats[std::size_t(corner.id)];
        const auto run = standing_at(seat);
        if (run.size() != 0)
          r = elect(run, seat);
        else {
          const auto k = mint_of_position[position_of(seat)];
          if (k == Index(-1)) {
            out.refusal = reseat_refusal::seat_unresolved;
            return out;
          }
          r = {Index(-1), out.minted_base + k};
        }
      }
      corners[c] = {std::int16_t(r[0]), r[1]};
    }
    out.piece_corners.push_back(corners);
  }
  // the delta world face of each piece, for its plane
  const auto n_promoted = std::size_t(n_pieces) + entered.size();
  tf::buffer<Index> piece_delta_face;
  piece_delta_face.allocate_and_initialize(n_promoted, Index(-1));
  for (Index face = 0; face < dw.n_faces(); ++face) {
    const auto &d = dw.descriptor(face);
    const auto &o = origin_of(Index(d.tag), d.object);
    if (o.role == delta_role::piece)
      piece_delta_face[std::size_t(o.id)] = face;
    else if (o.role == delta_role::restated) {
      const auto rank = entered_rank(o.tag, o.id);
      if (rank != Index(-1))
        piece_delta_face[std::size_t(n_pieces + rank)] = face;
    }
  }
  tf::buffer<tf::exact::canonical_plane<Int>> names;
  for (std::size_t k = 0; k < n_promoted; ++k) {
    if (piece_delta_face[k] == Index(-1)) {
      out.refusal = reseat_refusal::piece_unplaced;
      return out;
    }
    const auto &d = dw.descriptor(piece_delta_face[k]);
    names.push_back(tf::intersect::graph::name_plane_face<Index, Int>(
        Index(d.tag), d.object, dforms, dreader));
  }
  const auto n_world_planes = world.n_planes();
  if constexpr (tf::core::verification)
    tf::core::verify(Index(world_identities.local_plane_names().size()) ==
                         n_world_planes,
                     "the substrate names every plane the world holds");
  const auto standing_names = world_identities.plane_names();
  const auto n_named = Index(standing_names.size());
  tf::buffer<Index> new_plane_first;
  const auto n_new = tf::intersect::graph::seat_named_planes<Index, Int>(
      names, standing_names, n_named, out.piece_plane_of, new_plane_first);
  for (auto &plane : out.piece_plane_of) {
    if (plane < n_world_planes)
      continue;
    if (plane < n_named) {
      out.refusal = reseat_refusal::plane_promoted;
      return out;
    }
    plane -= n_named - n_world_planes;
  }
  out.new_frames.allocate(std::size_t(n_new));
  out.new_plane_names.allocate(std::size_t(n_new));
  for (Index r = 0; r < n_new; ++r) {
    const auto first = std::size_t(new_plane_first[std::size_t(r)]);
    out.new_frames[std::size_t(r)] =
        dw.frame(dw.plane_of_face(piece_delta_face[first]));
    out.new_plane_names[std::size_t(r)] = names[first];
  }
  out.piece_orientations.allocate(n_promoted);
  for (std::size_t k = 0; k < n_promoted; ++k) {
    const auto plane = out.piece_plane_of[k];
    const auto &frame =
        plane < n_world_planes
            ? world.frame(plane)
            : out.new_frames[std::size_t(plane - n_world_planes)];
    std::array<ref_t, 3> corners;
    if (k < std::size_t(n_pieces))
      for (std::size_t c = 0; c < 3; ++c)
        corners[c] = {Index(pieces.pieces[k].corners[c].tag),
                      pieces.pieces[k].corners[c].id};
    else {
      const auto &e = entered[k - std::size_t(n_pieces)];
      apply_to_form(e[0], [&](const auto &form) {
        const auto f = form.faces()[e[1]];
        for (std::size_t c = 0; c < 3; ++c)
          corners[c] = {e[0], Index(f[c])};
      });
    }
    out.piece_orientations[k] = tf::exact::plane_frame_winding(
        frame, std::array<std::size_t, 3>{0, 1, 2},
        [&](std::size_t c) { return point_of_ref(corners[c]); });
  }

  // a standing identity the carve's rows no longer
  // name, but a row the carve leaves standing still does, stood on a root the
  // delta split again; it moves to where the delta put it
  const auto state_retirements = [&] {
    std::sort(retirements.begin(), retirements.end());
    retirements.erase_till_end(
        std::unique(retirements.begin(), retirements.end()));
    tf::buffer<std::array<Index, 2>> welded;
    for (const auto &w : out.welds)
      welded.push_back({w[0], w[1]});
    std::sort(welded.begin(), welded.end());
    for (const auto &x : retirements)
      if (!std::binary_search(welded.begin(), welded.end(),
                              std::array<Index, 2>{x[0], x[1]}))
        out.welds.push_back(x);
  };
  // an identity an election retired moves where its retirement states; the
  // elections run on through the placement, so each judgment of what is
  // still unplaced reads the retirements as they stand when it is made
  const auto drop_retired = [&](tf::buffer<Index> &flats) {
    tf::buffer<Index> retired_flats;
    for (const auto &x : retirements)
      retired_flats.push_back(x[0] == Index(0) ? x[1] : n_flat + x[1]);
    std::sort(retired_flats.begin(), retired_flats.end());
    flats.erase_till_end(std::remove_if(
        flats.begin(), flats.end(), [&retired_flats](Index flat) {
          return std::binary_search(retired_flats.begin(), retired_flats.end(),
                                    flat);
        }));
  };
  tf::buffer<char> dying_face;
  dying_face.allocate_and_initialize(std::size_t(n_world_faces), char(0));
  for (const auto face : out.retired)
    dying_face[std::size_t(face)] = char(1);
  for (const auto face : out.restated)
    dying_face[std::size_t(face)] = char(1);
  tf::buffer<Index> alive;
  for (const auto &def : out.rows) {
    alive.push_back(flat_of({Index(def.point_tag_0), def.point_0}));
    alive.push_back(flat_of({Index(def.point_tag_1), def.point_1}));
  }
  std::sort(alive.begin(), alive.end());
  alive.erase_till_end(std::unique(alive.begin(), alive.end()));
  const auto is_alive = [&](Index flat) {
    return std::binary_search(alive.begin(), alive.end(), flat);
  };
  // the standing rows that stay: a dirty plane's rows of faces that live,
  // which are the only rows a carve's identities can share
  tf::buffer<Index> needed;
  const auto plane_rows = [&](auto &&visit) {
    for (const auto plane : planes)
      for (const auto row : world.plane_edges(plane))
        visit(world_defs[std::size_t(row)]);
  };
  plane_rows([&](const def_t &def) {
    if (dying_face[std::size_t(def.face)])
      return;
    const ref_t a{Index(def.point_tag_0), def.point_0};
    const ref_t b{Index(def.point_tag_1), def.point_1};
    for (const auto &r : {a, b}) {
      const auto flat = flat_of(r);
      if (!is_alive(flat))
        needed.push_back(flat);
    }
  });
  // only identities the dying rows named are candidates
  tf::buffer<Index> dying_named;
  plane_rows([&](const def_t &def) {
    if (!dying_face[std::size_t(def.face)])
      return;
    dying_named.push_back(flat_of({Index(def.point_tag_0), def.point_0}));
    dying_named.push_back(flat_of({Index(def.point_tag_1), def.point_1}));
  });
  std::sort(dying_named.begin(), dying_named.end());
  dying_named.erase_till_end(
      std::unique(dying_named.begin(), dying_named.end()));
  std::sort(needed.begin(), needed.end());
  needed.erase_till_end(std::unique(needed.begin(), needed.end()));
  // the standing identities are sorted already and the mints are few, so
  // one merge lays both on the order the band join reads
  const auto state_near = [&] {
    const auto window = world.near_window();
    if (window <= Int(0) || out.minted.size() == 0)
      return;
    using near_row_t =
        typename tf::intersect::graph::near_identity_rows<Index, pt3_t>::row_t;
    const auto row_less = [&point_less](const near_row_t &x,
                                        const near_row_t &y) {
      return point_less(x.point, y.point);
    };
    const auto row_of_flat = [n_flat](const pt3_t &point, Index flat) {
      return flat < n_flat
                 ? near_row_t{point,
                              Index(
                                  tf::intersect::graph::original_identity_rank),
                              flat}
                 : near_row_t{
                       point,
                       Index(tf::intersect::graph::standing_identity_rank),
                       flat - n_flat};
    };
    tf::buffer<near_row_t> minted_rows;
    minted_rows.allocate(out.minted.size());
    for (std::size_t k = 0; k < out.minted.size(); ++k)
      minted_rows[k] =
          row_of_flat(out.minted[k], n_flat + out.minted_base + Index(k));
    std::sort(minted_rows.begin(), minted_rows.end(), row_less);
    tf::buffer<near_row_t> standing_rows;
    standing_rows.allocate(identities.size());
    for (std::size_t k = 0; k < identities.size(); ++k)
      standing_rows[k] =
          row_of_flat(identities[k].point, flat_of(identities[k].ref));
    tf::buffer<near_row_t> rows;
    rows.allocate(standing_rows.size() + minted_rows.size());
    std::merge(standing_rows.begin(), standing_rows.end(), minted_rows.begin(),
               minted_rows.end(), rows.begin(), row_less);
    out.near_rows.append(std::move(rows), n_flat + out.minted_base);
  };
  tf::buffer<Index> moved;
  for (const auto flat : needed)
    if (std::binary_search(dying_named.begin(), dying_named.end(), flat))
      moved.push_back(flat);
  drop_retired(moved);
  // an original the delta's own gate joined to an identity speaks it there:
  // the rows the carve leaves standing follow it
  {
    tf::buffer<Index> kept;
    // the delta's created identities by lattice point, for the ones the
    // world names that the delta's gate moved
    struct dpos_t {
      pt3_t point;
      Index id;
    };
    tf::buffer<dpos_t> dpositions;
    for (const auto flat : moved)
      if (flat >= n_flat) {
        for (Index id = 0; id < dw_created; ++id)
          dpositions.push_back({dw.point_of(dw_identities, id, dreader), id});
        std::sort(dpositions.begin(), dpositions.end(),
                  [&](const dpos_t &x, const dpos_t &y) {
                    return point_less(x.point, y.point) ||
                           (!point_less(y.point, x.point) && x.id < y.id);
                  });
        break;
      }
    for (const auto flat : moved) {
      if (flat >= n_flat) {
        const auto p = point_of_ref({Index(-1), flat - n_flat});
        auto at = std::lower_bound(dpositions.begin(), dpositions.end(), p,
                                   [&](const dpos_t &x, const pt3_t &q) {
                                     return point_less(x.point, q);
                                   });
        Index target = Index(-1);
        for (; at != dpositions.end() && at->point == p && target == -1; ++at)
          target = dw_identities.local_target(Index(1), at->id);
        if (target != Index(-1)) {
          const auto r = resolve(std::int16_t(-1), target);
          const auto to = flat_of(r);
          if (to >= n_flat && to != flat) {
            out.welds.push_back({Index(1), flat - n_flat, to - n_flat});
            continue;
          }
        }
        kept.push_back(flat);
        continue;
      }
      const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
      const piece_corner<Index> corner{std::int16_t(tag),
                                       flat - vertex_offsets[std::size_t(tag)]};
      Index target = Index(-1);
      for (std::size_t t = 0; t < forms.vertices.size() && target == -1; ++t) {
        const auto &vs = forms.vertices[t];
        for (std::size_t v = 0; v < vs.size(); ++v)
          if (vs[v] == corner) {
            target = dw_identities.local_target(
                Index(0), dw.vertex_offsets()[t] + Index(v));
            break;
          }
      }
      if (target == Index(-1)) {
        kept.push_back(flat);
        continue;
      }
      const auto r = resolve(std::int16_t(-1), target);
      const auto to = flat_of(r);
      if (to < n_flat || to == flat) {
        kept.push_back(flat);
        continue;
      }
      out.welds.push_back({Index(0), flat, to - n_flat});
    }
    moved = std::move(kept);
  }
  drop_retired(moved);
  if (moved.size() == 0) {
    if (foreign) {
      out.refusal = reseat_refusal::name_foreign;
      return out;
    }
    state_retirements();
    state_near();
    out.ok = true;
    return out;
  }

  // the roots: a world root and the delta root over the same ends, each with
  // its chain of identities in order along it
  const auto world_roots = world.split_roots();
  const auto &wgraph = world.graph();
  const auto &dgraph = dw.graph();
  const auto droots = dw.split_roots();
  const auto wmerged = [&](const def_t &def) {
    const auto ends = tf::intersect::graph::plane_targeted_endpoints<Index>(
        world_name, vertex_offsets, def);
    ref_t lo{Index(ends.lo_tag), ends.lo_id};
    ref_t hi{Index(ends.hi_tag), ends.hi_id};
    if (hi < lo)
      std::swap(lo, hi);
    return std::array<Index, 4>{lo[0], lo[1], hi[0], hi[1]};
  };
  const auto dvertex_offsets = dw.vertex_offsets();
  const auto dmerged = [&](const def_t &def) {
    const auto ends = tf::intersect::graph::plane_merged_endpoints<Index>(
        dw_merges, dvertex_offsets, def);
    auto lo = resolve(ends.lo_tag, ends.lo_id);
    auto hi = resolve(ends.hi_tag, ends.hi_id);
    if (hi < lo)
      std::swap(lo, hi);
    return std::array<Index, 4>{lo[0], lo[1], hi[0], hi[1]};
  };
  struct root_t {
    std::array<Index, 4> key;
    std::size_t group;
  };
  tf::buffer<root_t> delta_root_keys;
  for (std::size_t g = 0; g < droots.size(); ++g)
    delta_root_keys.push_back({dmerged(dgraph.canon_group(droots[g])[0]), g});
  std::sort(delta_root_keys.begin(), delta_root_keys.end(),
            [](const root_t &x, const root_t &y) { return x.key < y.key; });
  for (std::size_t g = 0; g < world_roots.size(); ++g) {
    const auto survivors = world.split_survivors(g);
    tf::buffer<Index> dead;
    for (const auto id : survivors) {
      const auto flat = n_flat + id;
      if (std::binary_search(moved.begin(), moved.end(), flat))
        dead.push_back(flat);
    }
    if (dead.size() == 0)
      continue;
    const auto key = wmerged(wgraph.canon_group(world_roots[g])[0]);
    const auto at =
        std::lower_bound(delta_root_keys.begin(), delta_root_keys.end(), key,
                         [](const root_t &r, const std::array<Index, 4> &k) {
                           return r.key < k;
                         });
    if (at == delta_root_keys.end() || at->key != key) {
      out.refusal = reseat_refusal::root_unmatched;
      return out;
    }
    tf::buffer<Index> fresh;
    tf::buffer<Index> world_chain;
    for (const auto id : survivors)
      world_chain.push_back(n_flat + id);
    std::sort(world_chain.begin(), world_chain.end());
    for (const auto id : dw.split_survivors(at->group)) {
      const auto r = resolve(std::int16_t(-1), id);
      const auto flat = flat_of(r);
      if (!std::binary_search(world_chain.begin(), world_chain.end(), flat))
        fresh.push_back(flat);
    }
    if (fresh.size() != dead.size()) {
      out.refusal = reseat_refusal::root_survivors;
      return out;
    }
    for (std::size_t k = 0; k < dead.size(); ++k) {
      const auto source = dead[k];
      const auto target = fresh[k];
      if (target < n_flat) {
        out.refusal = reseat_refusal::root_onto_original;
        return out;
      }
      out.welds.push_back({source < n_flat ? Index(0) : Index(1),
                           source < n_flat ? source : source - n_flat,
                           target - n_flat});
    }
    for (const auto flat : dead)
      moved.erase_till_end(std::remove(moved.begin(), moved.end(), flat));
  }
  drop_retired(moved);
  if (moved.size() != 0) {
    out.refusal = reseat_refusal::moved_unplaced;
    return out;
  }
  if (foreign) {
    out.refusal = reseat_refusal::name_foreign;
    return out;
  }
  state_retirements();
  state_near();
  out.ok = true;
  return out;
}

} // namespace tf::arrangement::reseat
