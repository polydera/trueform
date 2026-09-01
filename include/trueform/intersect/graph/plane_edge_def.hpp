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

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::intersect::graph {

/// The instance's endpoints are stored in canonical key order; this bit
/// says its emission (base-loop) order is the other one.
inline constexpr std::uint8_t plane_edge_reversed_flag = 1;
/// The boundary instance covers its original edge whole.
inline constexpr std::uint8_t plane_edge_whole_side_flag = 2;
/// The instance carries the radial orientation of its producing pair's
/// carrier line (interior cuts only, and only where the generators
/// decide it).
inline constexpr std::uint8_t plane_edge_radial_flag = 4;
/// Set with @ref tf::intersect::graph::plane_edge_radial_flag when the
/// key-order direction runs AGAINST +(n_f x n_g).
inline constexpr std::uint8_t plane_edge_radial_reversed_flag = 8;
/// The instance is an intersection edge, and therefore a fan site:
/// planes meet here and a radial pairing stands. Born from an
/// intersection record, or a base-loop edge an intersection record
/// coincides with. Stated at extraction; coplanar-pair records state
/// nothing — a purely coplanar rim is never a fan.
inline constexpr std::uint8_t plane_edge_fan_flag = 16;
/// The instance's canonical pair carries more than two side
/// statements: the edge is the mesh's own non-manifold edge, where no
/// single continuation exists. Stated at canonicalization — welded
/// coincident edges keep distinct pairs and never add up.
inline constexpr std::uint8_t plane_edge_non_manifold_flag = 32;

/// One edge instance of a plane graph, stating everything about itself.
///
/// Endpoints are in key order — `(point_tag_0, point_0) <
/// (point_tag_1, point_1)` — so canonical identity is the record's own
/// leading key and every split parameter on the group is stated in one
/// frame; @ref tf::intersect::graph::plane_edge_reversed_flag recovers
/// the emission order. `id` is the canonical group (the span the record lies
/// in), `face` the group that emitted it, `ordinal` the base-loop position of
/// the loop-order start vertex (-1 for an interior cut), `side` the face-local
/// original edge the instance lies on (-1 for none). The plane carrying a row
/// is the carrier of its `face`, which the world answers.
template <typename Index> struct plane_edge_def {
  Index point_0;
  Index point_1;
  Index id;
  Index face;
  Index object_other;
  std::int16_t point_tag_0;
  std::int16_t point_tag_1;
  std::int16_t tag_other;
  std::int16_t ordinal;
  std::int16_t side;
  std::uint8_t flags;
};

/// CORE. The boundary definition one side of one face states: its two corners
/// in key order, covering the original edge WHOLE, with
/// @ref tf::intersect::graph::plane_edge_reversed_flag when the emission order
/// is the other one. The caller owns what the row is called (`id`), the group
/// that emits it (`face`) and the source face it lies on (`object_other`).
template <typename Index, typename Corners>
auto make_plane_boundary_side_def(std::int16_t tag, const Corners &corners,
                                  std::size_t side, Index id, Index face,
                                  Index object_other) -> plane_edge_def<Index> {
  std::array<Index, 2> a{Index(tag), Index(corners[side])};
  std::array<Index, 2> b{Index(tag),
                         Index(corners[(side + 1) % corners.size()])};
  std::uint8_t flags = plane_edge_whole_side_flag;
  if (b < a) {
    std::swap(a, b);
    flags = std::uint8_t(flags | plane_edge_reversed_flag);
  }
  return plane_edge_def<Index>{a[1],
                               b[1],
                               id,
                               face,
                               object_other,
                               std::int16_t(a[0]),
                               std::int16_t(b[0]),
                               tag,
                               std::int16_t(side),
                               std::int16_t(side),
                               flags};
}

/// The canonical key of a definition: its endpoints, already in key
/// order, are its whole identity.
template <typename Index>
auto plane_def_key_less(const plane_edge_def<Index> &x,
                        const plane_edge_def<Index> &y) -> bool {
  return std::tie(x.point_tag_0, x.point_0, x.point_tag_1, x.point_1) <
         std::tie(y.point_tag_0, y.point_0, y.point_tag_1, y.point_1);
}

/// The canonical group of a key-sorted table whose key the probe names,
/// `-1` when the table holds none: the groups ascend by key, so one
/// binary search states it.
template <typename Index, typename Tables>
auto find_plane_canon_group(const Tables &tables,
                            const plane_edge_def<Index> &probe) -> Index {
  Index lo = 0;
  Index hi = tables.n_canon();
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / Index(2);
    if (plane_def_key_less(tables.canon_group(mid)[0], probe))
      lo = mid + Index(1);
    else
      hi = mid;
  }
  return lo < tables.n_canon() &&
                 !plane_def_key_less(probe, tables.canon_group(lo)[0])
             ? lo
             : Index(-1);
}

/// The order a canonical group's rows stand in: the whole ORIENTED
/// instance — provenance then flags, because A->B and B->A on one key
/// are two facts. It is what a span is written in and what a search
/// into a span states; the key order above it is the group's own.
template <typename Index>
auto plane_def_instance_less(const plane_edge_def<Index> &x,
                             const plane_edge_def<Index> &y) -> bool {
  return std::tie(x.face, x.tag_other, x.object_other, x.ordinal, x.side,
                  x.flags) < std::tie(y.face, y.tag_other, y.object_other,
                                      y.ordinal, y.side, y.flags);
}

template <typename Index>
auto plane_edge_loop_start(const plane_edge_def<Index> &def)
    -> std::array<Index, 2> {
  return def.flags & plane_edge_reversed_flag
             ? std::array<Index, 2>{Index(def.point_tag_1), def.point_1}
             : std::array<Index, 2>{Index(def.point_tag_0), def.point_0};
}

template <typename Index>
auto plane_edge_loop_end(const plane_edge_def<Index> &def)
    -> std::array<Index, 2> {
  return def.flags & plane_edge_reversed_flag
             ? std::array<Index, 2>{Index(def.point_tag_0), def.point_0}
             : std::array<Index, 2>{Index(def.point_tag_1), def.point_1};
}

/// The radial orientation an instance carries: +1 when its key-order
/// direction runs along +(n_f x n_g) of its producing pair, -1 when it
/// runs against it, 0 when the instance carries no such fact.
template <typename Index>
auto plane_edge_radial_sign(const plane_edge_def<Index> &def) -> int {
  if (!(def.flags & plane_edge_radial_flag))
    return 0;
  return (def.flags & plane_edge_radial_reversed_flag) ? -1 : 1;
}

} // namespace tf::intersect::graph
