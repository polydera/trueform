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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/build_plane_edges.hpp"
#include "../../intersect/graph/canonicalize_plane_edge_defs.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/face_membership.hpp"
#include "../planes/plane_world.hpp"
#include "./build_mesh_plane_defs.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::arrangement {

/// THE MESH IS THE WORLD: one face is one plane carrier, its own boundary is
/// its constraint set, and a shared mesh edge is one canonical group — which
/// is what carries a split from a face to its neighbour.
///
/// The carrier space is arithmetic here — a plane IS its face and its only
/// member — so there is no extent to compare against and no suffix to branch
/// on. Nothing has been retired and no root is cut, so the split statement a
/// promoted side would inherit is empty; and a mesh world holds every face, so
/// there is nothing left to promote and the seam's `graph` is never asked for.
///
/// THE FRAME AND THE WINDING ARE ARITHMETIC ON THE FACE, and their consumer
/// runs at the carrier's own grain, so this world states them per carrier
/// instead of tabling them: a table written by one parallel pass and read by
/// the next at the SAME grain is pure loss. What the world does own is the
/// input's vertices on the lattice — read once per corner of every incident
/// face and again inside each triangulation, so that one IS a grain change.
///
/// THE DEFINITION TABLES ARE THE PRICE OF RESOLUTION, AND THIS WORLD PAYS IT
/// ONLY WHEN IT RESOLVES. A canonical group is a global sort's product and
/// exists to carry a split from one carrier to another; a mesh that states no
/// split never names one. So the tables start EMPTY — `materialized()` is the
/// whole statement of which of the two states this world is in — and a carrier
/// of an unmaterialized world reads the face it is. `materialize()` is the
/// one-way transition, and it is the barrier at which the canonical extent
/// becomes a real thing to freeze.
template <typename Index, typename Int, typename Faces> class plane_mesh_policy {
public:
  using index_type = Index;
  using tables_t = tf::intersect::graph::plane_tables<Index, Int>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;
  using frame_t = tf::exact::plane_frame<Int>;
  using point_t = tf::exact::pt3<Int>;

  /// The faces are a VIEW of the mesh this world answers for, so there is no
  /// empty world to fill in later: a world is constructed from its mesh.
  plane_mesh_policy(const Faces &faces, tf::buffer<point_t> &&points)
      : _faces(faces), _points(std::move(points)) {}

  auto tables() -> tables_t & { return _tables; }
  auto tables() const -> const tables_t & { return _tables; }

  auto n_planes() const -> Index { return Index(_faces.size()); }
  auto n_faces() const -> Index { return Index(_faces.size()); }
  auto member_count(Index) const -> Index { return Index(1); }
  auto member(Index plane, Index) const -> Index { return plane; }
  auto plane_of_face(Index face) const -> Index { return face; }

  /// THE CARRIER'S OWN BOUNDARY: a plane IS its face, so its corners are the
  /// whole statement of what bounds it. The world that can answer this is the
  /// world whose carriers are input faces, and it is the one fact the frame,
  /// the winding and the convex family all read.
  auto carrier_boundary(Index plane) const {
    return _faces[std::size_t(plane)];
  }
  /// One input vertex on the lattice this world quantized to.
  auto point(Index id) const -> const point_t & {
    return _points[std::size_t(id)];
  }
  auto n_points() const -> Index { return Index(_points.size()); }

  auto frame(Index plane) const -> frame_t {
    const auto corners = _faces[std::size_t(plane)];
    const auto &points = _points;
    return tf::exact::make_supported_plane_frame<Int>(
        [&corners, &points](const auto &consider) {
          for (const auto corner : corners)
            consider(points[std::size_t(corner)]);
        });
  }
  auto face_orientation(Index face) const -> std::int8_t {
    const auto &points = _points;
    return tf::exact::plane_frame_winding(frame(face), _faces[std::size_t(face)],
                                          [&points](auto corner) {
                                            return points[std::size_t(corner)];
                                          });
  }
  auto descriptor(Index face) const -> descriptor_t {
    return {Index(0), face};
  }
  auto descriptors() const {
    return tf::make_mapped_range(
        tf::make_sequence_range(Index(_faces.size())),
        [](Index face) -> descriptor_t { return {Index(0), face}; });
  }

  auto edge_defs() const { return _tables.edge_defs(); }
  auto n_canon() const -> Index { return _tables.n_canon(); }
  auto canon_group(Index canon) const { return _tables.canon_group(canon); }
  auto plane_edges(Index plane) const { return _tables.plane_edges(plane); }

  /// Whether the definition tables are real. Unmaterialized, this world has
  /// stated nothing about a carrier but the face it is.
  auto materialized() const -> bool { return _tables.edges().size() != 0; }

  /// THE ONE-WAY TRANSITION: the definitions the mesh states, canonicalized
  /// into the group space a split is carried in, the plane CSR over them, and
  /// the membership `carriers_of_flat` answers by. A plane's face list is the
  /// face itself, so the CSR the closure builds is that face's own rows.
  ///
  /// This is the barrier a wave needs and nothing else does, so a build that
  /// states no split and no weld never runs it.
  auto materialize() -> void {
    if (materialized() || _faces.size() == 0)
      return;
    _membership.build(_faces, _points.size());
    const auto n_faces = Index(_faces.size());
    // one face is one plane and its only member, so the carrier lookups the
    // graph's producers read are arithmetic here
    const auto plane_faces =
        tf::make_blocked_range<1>(tf::make_sequence_range(n_faces));
    tf::buffer<Index> face_def_offsets;
    build_mesh_plane_defs<Index>(_faces, _tables.defs(), face_def_offsets);
    tf::buffer<Index> emission_to_canon;
    tf::intersect::graph::canonicalize_plane_edge_defs<Index>(
        _tables.defs(), _tables.def_offsets(), _tables.n_canon(),
        emission_to_canon);
    tf::intersect::graph::build_plane_edges<Index>(
        n_faces, plane_faces, face_def_offsets, emission_to_canon,
        _tables.edges());
  }

  /// THE CARRIERS HOLDING ONE FLAT IDENTITY. A flat below the input's own
  /// extent IS a mesh vertex id and a plane IS its face, so the carriers are
  /// the faces that name that vertex — one CSR lookup, never a scan. A created
  /// identity is no vertex of this world and answers with nothing, which is
  /// exactly what @ref tf::arrangement::states_flat_carriers asks of a world
  /// that answers at all.
  auto carriers_of_flat(Index flat) const {
    return flat >= Index(0) && flat < Index(_points.size()) &&
                   _membership.size() != 0
               ? _membership[std::size_t(flat)]
               : tf::range<const Index *, tf::dynamic_size>(nullptr, nullptr);
  }

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

private:
  Faces _faces;
  tf::buffer<point_t> _points;
  tables_t _tables;
  tf::face_membership<Index> _membership;
};

template <typename Index, typename Int, typename Faces>
using plane_mesh_world = plane_world<plane_mesh_policy<Index, Int, Faces>>;

/// The world a mesh states: its faces and its vertices on the lattice, and
/// nothing else. The definition tables it would need to carry a split are not
/// built here — @ref tf::arrangement::plane_mesh_policy::materialize states
/// them at the barrier that first needs one.
template <typename Index, typename Int, typename Faces, typename GetPoint>
auto make_plane_mesh_world(const Faces &faces, Index n_points,
                           const GetPoint &get_point)
    -> plane_mesh_world<Index, Int, Faces> {
  tf::buffer<tf::exact::pt3<Int>> points;
  points.allocate(std::size_t(n_points));
  tf::parallel_for_each(
      tf::make_sequence_range(n_points),
      [&points, &get_point](Index id) {
        points[std::size_t(id)] = get_point(id);
      },
      tf::checked);
  return plane_mesh_world<Index, Int, Faces>(faces, std::move(points));
}

} // namespace tf::arrangement
