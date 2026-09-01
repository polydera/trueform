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

#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/local_arrangement.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// THE PREPARED WORLD one plane arrangement is built on, and the ONE
/// carrier its seam speaks. The policy owns the storage; composition is
/// compile time and costs nothing.
///
/// THE CONTRACT, STATED ONCE. A policy answers these names and the tier reads
/// nothing else, so a new policy implements a stated concept instead of
/// re-negotiating a seam. Each name carries its GRAIN — when it is read — and
/// its LAZINESS LICENSE — whether it may be computed on demand, or must be
/// frozen for the whole build.
///
/// ENTRY GRAIN, FROZEN. Read once, when the build initializes, and stated in
/// every ticket published afterwards, so they may not move while a build runs:
/// `n_planes` and `n_faces`, the carrier and face extents, and `n_canon`, the
/// world's canonical extent — the group space's frontier. A world that defers
/// its definition tier states `n_canon` at the barrier that makes the group
/// space real instead: before that barrier nothing this arrangement holds is
/// expressed in the group space, so freezing there is the FIRST statement of
/// the extent and not a second.
///
/// ROUND GRAIN, ON DEMAND. Read inside the per-carrier iteration that consumes
/// them and by nothing else, so a policy may answer them when asked and keep
/// nothing: `tables` — the world's definition tier, its rows, its canonical
/// groups (`edge_defs`, `canon_group`, `plane_edges`) and the block that names
/// one carrier's constraint set, which the wave ports FROM and never writes;
/// `frame`; `member_count` and `member`, the faces one carrier holds in member
/// order (`plane_members` composes them); `plane_of_face`, `descriptor`,
/// `descriptors` and `face_orientation`. And — OPTIONAL — `carrier_boundary`,
/// the corners of the face a carrier IS, which a world whose definition tier
/// is not yet real states its constraint set with instead of a block.
///
/// THE TWO STATES OF A DEFINITION TIER: `materialized` says whether the
/// world's own tables are real, and `materialize` is the one-way transition to
/// them. The tables carry a split from one carrier to another, so a world
/// whose carrier IS an input face can answer every round-grain name without
/// them and builds them at the first refusal or weld — the only events that
/// need a group space. A world born with its tables answers `true` and its
/// transition is empty, which is the whole cost of the license.
///
/// WAVE GRAIN, ON DEMAND, reached only when a round refuses or welds:
/// `canon_group`, a root's instances, and — OPTIONAL —
/// `carriers_of_flat`, which carriers hold one flat identity;
/// @ref tf::arrangement::states_flat_carriers states what answering it commits
/// a world to, and a world that does not answer it is swept whole.
///
/// THE IDENTITY SPACE IS THE ENTRY'S PARAMETER, not a name every world states:
/// the closed-world entry is given `n_base_created` and `vertex_offsets` at the
/// boundary. A world built ON a local arrangement carries that space itself —
/// `vertex_offsets`, `face_offsets`, `n_created_points`, `point_of`,
/// `intersection_points` — and the refinement's entrant side reads its split
/// statement (`merges`, `split_roots`, `split_survivors`) and its `graph`.
///
/// A policy that can be promoted states the barrier as a CONSTRUCTOR, so a
/// promoted world is a new VALUE of the same type and the wave reads one
/// world type whether or not anything was promoted.
template <typename Policy> struct plane_world : public Policy {
  using Policy::Policy;

  /// The faces one plane carries, in member order.
  auto plane_members(typename Policy::index_type plane) const {
    using index_type = typename Policy::index_type;
    return tf::make_mapped_range(
        tf::make_sequence_range(Policy::member_count(plane)),
        [this, plane](index_type position) {
          return this->Policy::member(plane, position);
        });
  }
};

/// The production policy: the local arrangement IS the world, borrowed, and
/// past its extents the refinement's own promoted rows.
///
/// One policy states BOTH modes: a stock build carries an EMPTY extension, so
/// its suffix branch is never taken. `local_arrangement` stitches its own cut
/// and entrant spaces behind the same compare.
///
/// THE EXTENTS ARE VALUES, NOT DERIVATIONS. The promotion states its extent
/// once, in the constructor that builds the promoted value, so a reader still
/// holding the base value keeps the base extents by construction and no phase
/// order can make an extent mean two things.
template <typename Index, typename RealType, typename Int>
class plane_local_arrangement_policy {
public:
  using index_type = Index;
  using arrangement_t =
      tf::intersect::graph::local_arrangement<Index, RealType, Int>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;

  explicit plane_local_arrangement_policy(const arrangement_t &arrangement)
      : _arrangement(&arrangement), _promoted_descriptors(nullptr, nullptr),
        _promoted_frames(nullptr, nullptr),
        _promoted_orientations(nullptr, nullptr),
        _n_base_planes(arrangement.n_planes()),
        _n_base_faces(arrangement.n_faces()),
        _n_planes(arrangement.n_planes()), _n_faces(arrangement.n_faces()) {}

  /// THE PROMOTION BARRIER: the base value, extended by the rows the
  /// arrangement itself named. Every suffix plane's edge definitions already
  /// live in the arrangement's local table, so the base tables remain the
  /// world authority and a suffix plane must hold a local ticket.
  ///
  /// The promotion appends one plane and one face per row, so the descriptor
  /// count is the one extent both spaces grow by.
  template <typename Descriptors, typename Frames, typename Orientations>
  plane_local_arrangement_policy(const plane_local_arrangement_policy &base,
                                 const Descriptors &descriptors,
                                 const Frames &frames,
                                 const Orientations &orientations)
      : _arrangement(base._arrangement),
        _promoted_descriptors(descriptors.begin(), descriptors.end()),
        _promoted_frames(frames.begin(), frames.end()),
        _promoted_orientations(orientations.begin(), orientations.end()),
        _n_base_planes(base._n_base_planes), _n_base_faces(base._n_base_faces),
        _n_planes(base._n_base_planes + Index(descriptors.size())),
        _n_faces(base._n_base_faces + Index(descriptors.size())) {}

  auto graph() const -> const auto & { return _arrangement->graph(); }
  auto tables() const -> const auto & { return _arrangement->tables(); }
  /// The local arrangement states its tier before this world borrows it, so
  /// the transition is already behind and costs nothing to ask for.
  auto materialized() const -> bool { return true; }
  auto materialize() -> void {}
  auto edge_defs() const { return _arrangement->edge_defs(); }
  auto n_canon() const -> Index { return _arrangement->n_canon(); }
  auto canon_group(Index canon) const {
    return _arrangement->canon_group(canon);
  }
  auto plane_edges(Index plane) const {
    return _arrangement->plane_edges(plane);
  }
  auto merges() const { return _arrangement->merges(); }
  auto split_roots() const { return _arrangement->split_roots(); }
  auto split_survivors(std::size_t group) const {
    return _arrangement->split_survivors(group);
  }

  auto n_planes() const -> Index { return _n_planes; }
  auto n_faces() const -> Index { return _n_faces; }
  auto frame(Index plane) const -> const frame_t & {
    return plane < _n_base_planes
               ? _arrangement->frame(plane)
               : _promoted_frames[std::size_t(plane - _n_base_planes)];
  }
  auto member_count(Index plane) const -> Index {
    return plane < _n_base_planes ? _arrangement->member_count(plane)
                                  : Index(1);
  }
  auto member(Index plane, Index position) const -> Index {
    return plane < _n_base_planes
               ? _arrangement->member(plane, position)
               : _n_base_faces + (plane - _n_base_planes);
  }
  auto plane_of_face(Index face) const -> Index {
    return face < _n_base_faces ? _arrangement->plane_of_face(face)
                                : _n_base_planes + (face - _n_base_faces);
  }
  auto descriptor(Index face) const -> const descriptor_t & {
    return face < _n_base_faces
               ? _arrangement->descriptor(face)
               : _promoted_descriptors[std::size_t(face - _n_base_faces)];
  }
  auto descriptors() const {
    const auto base = _arrangement->descriptors();
    const auto promoted = _promoted_descriptors;
    const auto base_faces = _n_base_faces;
    return tf::make_mapped_range(
        tf::make_sequence_range(_n_faces),
        [base, promoted, base_faces](Index face) -> const descriptor_t & {
          return face < base_faces
                     ? base[std::size_t(face)]
                     : promoted[std::size_t(face - base_faces)];
        });
  }
  auto face_orientation(Index face) const -> std::int8_t {
    return face < _n_base_faces
               ? _arrangement->face_orientation(face)
               : _promoted_orientations[std::size_t(face - _n_base_faces)];
  }

  auto vertex_offsets() const { return _arrangement->vertex_offsets(); }
  auto face_offsets() const { return _arrangement->face_offsets(); }
  auto n_created_points() const -> Index {
    return _arrangement->n_created_points();
  }
  auto intersection_points() const {
    return _arrangement->intersection_points();
  }
  template <typename GetPoint, typename GetMeshPoint>
  auto point_of(Index id, const GetPoint &get_point,
                const GetMeshPoint &get_mesh_point) const {
    return _arrangement->point_of(id, get_point, get_mesh_point);
  }

private:
  const arrangement_t *_arrangement;
  tf::range<const descriptor_t *, tf::dynamic_size> _promoted_descriptors;
  tf::range<const frame_t *, tf::dynamic_size> _promoted_frames;
  tf::range<const std::int8_t *, tf::dynamic_size> _promoted_orientations;
  Index _n_base_planes;
  Index _n_base_faces;
  Index _n_planes;
  Index _n_faces;
};

template <typename Index, typename RealType, typename Int>
auto make_plane_world(
    const tf::intersect::graph::local_arrangement<Index, RealType, Int>
        &arrangement)
    -> plane_world<plane_local_arrangement_policy<Index, RealType, Int>> {
  return plane_world<plane_local_arrangement_policy<Index, RealType, Int>>(
      arrangement);
}

} // namespace tf::arrangement
