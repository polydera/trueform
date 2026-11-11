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
#include "../../arrangement/identity/identity_substrate.hpp"
#include "../../arrangement/planes/plane_arrangement.hpp"
#include "../../arrangement/planes/plane_world.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/build_plane_edges.hpp"
#include "../../intersect/graph/canonicalize_plane_edge_defs.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/make_plane_face_frame.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_face_support.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./state_tet_lane_occurrences.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::tetmesh {

/// THE DRAIN'S NAMES IN THE ARRANGEMENT'S CURRENCY: the authority's
/// originals are the flat vertices, `(0, point)`, and the identities the
/// drains created are the arrangement's base created identities,
/// `(-1, point - n_originals)`, standing below every identity a wave mints —
/// so a flat identity below `n_originals + n_created` IS the authority's
/// identity of that number, and the closure's weld window joins a wave's
/// mint to a created identity standing in its box, the standing one
/// surviving, as it does an arrangement's local tier. `created_name` names
/// the wave's own identities, past the base.
template <typename Index> struct tet_drain_names {
  Index n_originals;
  Index n_created;

  auto name(Index point) const -> std::pair<std::int16_t, Index> {
    return point < n_originals
               ? std::pair<std::int16_t, Index>{std::int16_t(0), point}
               : std::pair<std::int16_t, Index>{std::int16_t(-1),
                                                Index(point - n_originals)};
  }
  /// The authority's identity of created identity `id` of the arrangement.
  auto point_of_created(Index id, const tf::buffer<Index> &created_name) const
      -> Index {
    return id < n_created ? Index(n_originals + id)
                          : created_name[std::size_t(id - n_created)];
  }
  auto point(std::int16_t tag, Index id,
             const tf::buffer<Index> &created_name) const -> Index {
    return tag >= 0 ? id : point_of_created(id, created_name);
  }
  auto point_of_flat(Index flat, const tf::buffer<Index> &created_name) const
      -> Index {
    return flat < n_originals
               ? flat
               : point_of_created(flat - n_originals, created_name);
  }
};

/// THE DRAIN IS THE WORLD: the plane arrangement's world over one drain's
/// lanes. A carrier is a lane — the triangles coplanar with one another, one
/// exact plane, or one triangle — and its members are those triangles, the
/// face a member names being its position in `members`. The identity space
/// is the surface's own points in the names @ref tet_drain_names states, so
/// the plane layer holds only what the wave creates.
///
/// The definition tier is the drain's statement — every member's sides,
/// split at the points the drain put on them, and every chord's pieces — so
/// the world is born with its tables: a side two lanes hold and a chord the
/// two faces of its pair hold are one canonical group each, which is what
/// carries a split the wave states into every carrier that holds it. Nothing
/// is retired and no root is cut, so the split statement a promoted side
/// would inherit is empty, and the world holds every lane a split can reach,
/// so nothing is promoted and the seam's `graph` is never asked for.
template <typename Index, typename Int, typename Points>
class plane_drain_policy {
  using lanes_t = tet_lane_occurrences<Index, Int>;

public:
  using index_type = Index;
  using tables_t = tf::intersect::graph::plane_tables<Index, Int>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;
  using point_t = tf::exact::pt3<Int>;

  plane_drain_policy(const Points &points,
                     const tf::buffer<Index> &member_offsets,
                     const tf::buffer<std::array<Index, 3>> &members,
                     const tables_t &tables, const lanes_t &lanes)
      : _points(&points), _member_offsets(&member_offsets), _members(&members),
        _tables(&tables), _lanes(&lanes) {}

  auto tables() const -> const tables_t & { return *_tables; }

  auto n_planes() const -> Index { return Index(_member_offsets->size() - 1); }
  auto n_faces() const -> Index { return Index(_members->size()); }
  auto member_count(Index plane) const -> Index {
    return (*_member_offsets)[std::size_t(plane) + 1] -
           (*_member_offsets)[std::size_t(plane)];
  }
  auto member(Index plane, Index position) const -> Index {
    return (*_member_offsets)[std::size_t(plane)] + position;
  }
  auto plane_of_face(Index face) const -> Index {
    return Index(std::upper_bound(_member_offsets->begin() + 1,
                                  _member_offsets->end(), face) -
                 (_member_offsets->begin() + 1));
  }

  auto point(Index id) const -> point_t {
    const auto p = (*_points)[std::size_t(id)];
    return point_t{p[0], p[1], p[2]};
  }
  /// A lane's chart reads a point at its occurrence there
  /// (@ref state_tet_lane_occurrences), every other tier at its one position.
  auto chart_point(Index plane, Index flat, const point_t &own) const
      -> point_t {
    return _lanes->chart(plane, flat, own);
  }

  auto frame(Index plane) const -> frame_t {
    const auto &corners = (*_members)[std::size_t(member(plane, Index(0)))];
    return tf::intersect::graph::make_plane_face_frame<Int>(
        corners, [&](std::size_t corner) { return point(corners[corner]); });
  }
  auto face_orientation(Index face) const -> std::int8_t {
    return face_orientation(face, frame(plane_of_face(face)));
  }
  auto face_orientation(Index face, const frame_t &frame) const -> std::int8_t {
    return tf::exact::plane_frame_winding(
        frame, (*_members)[std::size_t(face)],
        [this](auto corner) { return point(Index(corner)); });
  }
  auto descriptor(Index face) const -> descriptor_t { return {Index(0), face}; }
  auto descriptors() const {
    return tf::make_mapped_range(
        tf::make_sequence_range(n_faces()),
        [](Index face) -> descriptor_t { return {Index(0), face}; });
  }

  auto edge_defs() const { return _tables->edge_defs(); }
  auto n_canon() const -> Index { return _tables->n_canon(); }
  auto canon_group(Index canon) const { return _tables->canon_group(canon); }
  auto plane_edges(Index plane) const { return _tables->plane_edges(plane); }
  auto materialized() const -> bool { return true; }
  auto materialize() -> void {}

  auto merges() const {
    return tf::range<const std::array<Index, 3> *, tf::dynamic_size>(nullptr,
                                                                     nullptr);
  }
  auto split_roots() const {
    return tf::range<const Index *, tf::dynamic_size>(nullptr, nullptr);
  }
  auto split_survivors(std::size_t) const {
    return tf::range<const Index *, tf::dynamic_size>(nullptr, nullptr);
  }
  auto plane_name(Index plane) const -> tf::exact::canonical_plane<Int> {
    const auto &corners = (*_members)[std::size_t(member(plane, Index(0)))];
    return tf::exact::make_canonical_plane<Int>(
        tf::intersect::graph::plane_face_support<Int>(
            corners,
            [&](std::size_t corner) { return point(corners[corner]); }));
  }
  auto source_side(Index face, Index side) const -> std::array<Index, 2> {
    return {face, side};
  }

private:
  const Points *_points;
  const tf::buffer<Index> *_member_offsets;
  const tf::buffer<std::array<Index, 3>> *_members;
  const tables_t *_tables;
  const lanes_t *_lanes;
};

template <typename Index, typename Int, typename Points>
using plane_drain_world =
    tf::arrangement::plane_world<plane_drain_policy<Index, Int, Points>>;

/// The drain's definition tier into `tables`, its endpoints in `names`: per
/// member, in member order,
/// one boundary row per piece of each of its sides — `side_points(f, k,
/// out)` filling the points on side `k` of member `f` in order from its
/// corner `k` — and one interior row per chord piece `chord_pieces(f, out)`
/// names, canonicalized into the groups a split is carried in and the plane
/// CSR over the lanes `member_offsets` blocks.
template <typename Index, typename Int, typename SidePoints,
          typename ChordPieces>
auto state_tet_drain_tables(
    const tet_drain_names<Index> &names,
    const tf::buffer<Index> &member_offsets,
    const tf::buffer<std::array<Index, 3>> &members,
    const SidePoints &side_points, const ChordPieces &chord_pieces,
    tf::buffer<Index> &face_def_offsets, tf::buffer<Index> &emission_to_canon,
    tf::buffer<Index> &run, tf::buffer<std::array<Index, 2>> &pieces,
    tf::intersect::graph::plane_tables<Index, Int> &tables) -> void {
  namespace graph = tf::intersect::graph;
  auto &defs = tables.defs();
  defs.clear();
  const std::size_t n_faces = members.size();
  face_def_offsets.allocate(n_faces + 1);
  face_def_offsets[0] = Index(0);
  const auto push = [&defs, &names](Index a, Index b, Index face,
                                    std::int16_t ordinal, std::int16_t side,
                                    bool whole) {
    std::uint8_t flags =
        whole ? graph::plane_edge_whole_side_flag : std::uint8_t(0);
    auto from = names.name(a);
    auto to = names.name(b);
    if (to < from) {
      std::swap(from, to);
      flags = std::uint8_t(flags | graph::plane_edge_reversed_flag);
    }
    defs.push_back(graph::plane_edge_def<Index>{
        from.second, to.second, Index(defs.size()), face, face, from.first,
        to.first, std::int16_t(0), ordinal, side, flags});
  };
  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto &corners = members[f];
    std::int16_t ordinal = 0;
    for (std::size_t k = 0; k < 3; ++k) {
      side_points(f, k, run);
      Index from = corners[k];
      for (auto p : run) {
        push(from, p, Index(f), ordinal++, std::int16_t(k), false);
        from = p;
      }
      push(from, corners[(k + 1) % 3], Index(f), ordinal++, std::int16_t(k),
           run.size() == 0);
    }
    chord_pieces(f, pieces);
    for (const auto &piece : pieces)
      push(piece[0], piece[1], Index(f), std::int16_t(-1), std::int16_t(-1),
           false);
    face_def_offsets[f + 1] = Index(defs.size());
  }
  graph::canonicalize_plane_edge_defs<Index>(defs, tables.def_offsets(),
                                             emission_to_canon);
  graph::build_plane_edges<Index>(
      Index(member_offsets.size() - 1),
      tf::make_offset_block_range(member_offsets,
                                  tf::make_sequence_range(Index(n_faces))),
      face_def_offsets, emission_to_canon, tables.edges());
}

/// The wave over the drain's `world` in its `names`: the authority's
/// originals the flat vertices of `vertex_offsets`, its created identities
/// the base created extent, and the identities the wave mints in one box of
/// side `reach` one with each other and with a created identity standing
/// there (@ref tf::arrangement::plane_arrangement::weld_near_identities).
/// `read(tag, id)` reads a name's position.
template <typename Index, typename Int, typename World, typename Read>
auto arrange_tet_drain(
    World &world, const tet_drain_names<Index> &names, Int reach,
    const Read &read, tf::buffer<Index> &vertex_offsets,
    tf::arrangement::identity_substrate<Index, Int> &identities,
    tf::arrangement::plane_arrangement<Index, Int> &arrangement) -> void {
  vertex_offsets.allocate(2);
  vertex_offsets[0] = Index(0);
  vertex_offsets[1] = names.n_originals;
  arrangement.weld_near_identities(reach);
  arrangement.build(world, identities, names.n_created, read, read,
                    vertex_offsets);
}

} // namespace tf::tetmesh
