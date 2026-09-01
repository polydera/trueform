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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../exact/plane_frame.hpp"
#include "./build_plane_csr.hpp"
#include "./build_plane_edge_defs.hpp"
#include "./build_plane_edges.hpp"
#include "./build_plane_frames.hpp"
#include "./build_plane_loops.hpp"
#include "./build_plane_orientations.hpp"
#include "./canonicalize_plane_edge_defs.hpp"
#include "./face_descriptor.hpp"
#include "./name_plane_carriers.hpp"
#include "./plane_edge_def.hpp"
#include "./vertex.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// The plane graph: one canon-major table of edge definitions, the
/// planes their faces pool into, and each plane's edge set.
///
/// The definition table is sorted by canonical identity, so a group is
/// a span of it and `_edge_defs_offsets` is the whole reverse lookup —
/// no instance list, no representative table. Because a definition
/// names its own face, base-loop position and original edge, the base
/// loops are not a carrier either: a face's boundary definitions ARE
/// its loop, in ordinal order. What only the loops can answer — the
/// plane's projection frame and each member's winding in it — is
/// answered here, once, while they are still in hand.
template <typename Index, typename Int> class plane_graph {
public:
  using vertex_t = vertex<Index>;
  using def_t = plane_edge_def<Index>;
  using frame_t = tf::exact::plane_frame<Int>;

  /// The faces this graph names, ASCENDING BY `(tag, object)`: they are
  /// the face carrier's own blocks, aggregated in input order, so the flat
  /// source key ascends with them and a consumer answers "does the cut
  /// world name this face" by one binary search.
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto edge_defs() { return tf::make_range(_edge_defs); }
  auto edge_defs() const { return tf::make_range(_edge_defs); }
  auto n_canon() const -> Index { return _n_canon; }
  auto canon_group(Index id) const {
    return tf::slice(tf::make_range(_edge_defs),
                     std::size_t(_edge_defs_offsets[std::size_t(id)]),
                     std::size_t(_edge_defs_offsets[std::size_t(id) + 1]));
  }
  auto n_planes() const -> Index { return _n_planes; }
  auto plane_of() const { return tf::make_range(_plane_of); }
  auto plane_faces() const {
    return tf::make_offset_block_range(_plane_offsets,
                                       tf::make_range(_plane_data));
  }
  auto plane_faces(Index p) const {
    return tf::slice(tf::make_range(_plane_data),
                     std::size_t(_plane_offsets[std::size_t(p)]),
                     std::size_t(_plane_offsets[std::size_t(p) + 1]));
  }
  auto plane_edges() const { return tf::make_range(_edges); }
  auto plane_edges(Index p) const {
    return tf::make_range(_edges)[std::size_t(p)];
  }
  auto frame(Index p) const -> const frame_t & {
    return _frames[std::size_t(p)];
  }
  auto face_orientation() const { return tf::make_range(_face_orientation); }
  auto face_contour() const { return tf::make_range(_face_contour); }

  auto clear() -> void {
    _descriptors.clear();
    _edge_defs.clear();
    _edge_defs_offsets.clear();
    _n_canon = 0;
    _edges.clear();
    _plane_of.clear();
    _n_planes = 0;
    _plane_offsets.clear();
    _plane_data.clear();
    _frames.clear();
    _face_orientation.clear();
    _face_contour.clear();
  }

  template <typename Ibp, typename ApplyToFace, typename ApplyToForm,
            typename GetPoint, typename SideNonManifold>
  auto build(const Ibp &ibp, const ApplyToFace &apply_to_face,
             const ApplyToForm &apply_to_form, const GetPoint &get_point,
             const SideNonManifold &side_non_manifold) -> void {
    clear();
    // ONE carrier, two currencies at the same block position: a face is
    // here because a pair record names it or because a delivery does, and
    // either block may be empty
    const auto subranges = ibp.intersections();
    const auto deliveries = ibp.deliveries();
    if (subranges.size() == 0)
      return;
    tf::offset_block_buffer<Index, vertex_t> loops;
    build_plane_loops<Index, Int>(subranges, deliveries, apply_to_face,
                                  get_point, _descriptors, loops);
    tf::buffer<Index> face_def_offsets;
    build_plane_edge_defs<Index, Int>(
        tf::make_range(_descriptors), subranges, tf::make_range(loops), ibp,
        apply_to_face, get_point, side_non_manifold, _edge_defs, _face_contour,
        face_def_offsets);
    // THE IDENTITY IS THE NAME. Every carrier names the plane it stands
    // on and equal names are one plane, so the identity is settled before
    // a single definition is read and nothing closes a pairwise relation
    // to reach it.
    name_plane_carriers<Index, Int>(_descriptors, apply_to_form, get_point,
                                    _plane_of, _n_planes);
    build_plane_csr<Index>(_plane_of, _n_planes, _plane_offsets, _plane_data);
    build_plane_frames<Index, Int>(_n_planes, plane_faces(), _descriptors,
                                   apply_to_face, get_point, _frames);
    build_plane_orientations<Index, Int>(_descriptors, _plane_of, _frames,
                                         apply_to_face, get_point,
                                         _face_orientation);
    tf::buffer<Index> emission_to_canon;
    canonicalize_plane_edge_defs<Index>(_edge_defs, _edge_defs_offsets,
                                        _n_canon, emission_to_canon);
    build_plane_edges<Index>(_n_planes, plane_faces(), face_def_offsets,
                             emission_to_canon, _edges);
  }

private:
  tf::buffer<face_descriptor<Index>> _descriptors;
  // canon-major: a canonical group is a span of the table
  tf::buffer<def_t> _edge_defs;
  tf::buffer<Index> _edge_defs_offsets;
  // plane -> its members' edge instances: THE working carrier's edge
  // set, pooled members' duplicates included by construction
  tf::offset_block_buffer<Index, Index> _edges;
  tf::buffer<Index> _plane_of; // dense plane id per face group
  Index _n_planes = 0;
  tf::buffer<Index> _plane_offsets;
  tf::buffer<Index> _plane_data;
  Index _n_canon = 0;
  // per plane: the projection its arrangement is computed in
  tf::buffer<frame_t> _frames;
  // per face group: its winding in its plane's projection
  tf::buffer<std::int8_t> _face_orientation;
  // per face group: -1 = no interior contour edges, -2 = several
  // distinct contour tags, else the single tag_other
  tf::buffer<std::int16_t> _face_contour;
};

} // namespace tf::intersect::graph
