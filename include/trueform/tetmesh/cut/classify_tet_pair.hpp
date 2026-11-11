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
#include "../../core/aabb.hpp"
#include "../../core/range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/classify/face_plane_info.hpp"
#include "../../intersect/face_pairs/certify_shared_feature_pair.hpp"
#include "../../intersect/face_pairs/face_pair_search.hpp"
#include "../../intersect/face_pairs/face_self_kernels.hpp"
#include "../../intersect/face_pairs/keep_face_pair_block.hpp"
#include "../../intersect/face_pairs/placed_face_box.hpp"
#include "../../topology/topo_type.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::tetmesh {

/// The scratch @ref classify_tet_pair walks on.
template <typename Index, typename Int> struct tet_pair_kernel_workspace {
  tf::intersect::face_pair_workspace<Index, Int> pair;
  std::array<tf::exact::vertex<Index, Int>, 6> corners{};
};

/// The contacts of the triangles `pair` (`t0 < t1`) of the lattice
/// `points`, `triangles` naming their corners, as the intersect module's
/// pair kernel states them for the two triangles read as one form (@ref
/// tf::intersect::within_polygon_pair_prepped behind its box test and @ref
/// tf::intersect::certify_shared_feature_pair): its records land in
/// `work.pair.intersections`, `object` the side, `target` where the contact
/// stands on it, and a pair that meets holds every corner both share as a
/// contact of the two, as the form's build delivers it. The form is never
/// built — its points are numbered by first
/// appearance and its gating is the two-face form's own: a corner both
/// triangles hold is the second triangle's, an edge both hold the first's.
/// Returns whether the triangles meet. O(1).
template <typename Index, typename Int, typename Points, typename Triangles>
auto classify_tet_pair(const Points &points, const Triangles &triangles,
                       const std::array<Index, 2> &pair,
                       tet_pair_kernel_workspace<Index, Int> &work) -> bool {
  auto &ws = work.pair;
  ws.intersections.clear();
  ws.payloads.clear();
  ws.coplanar_pairs.clear();
  std::array<std::array<Index, 3>, 2> ids{};
  std::array<Index, 6> named{};
  std::size_t n_named = 0;
  for (std::size_t s = 0; s < 2; ++s)
    for (std::size_t k = 0; k < 3; ++k) {
      const Index v = Index(triangles[std::size_t(pair[s])][k]);
      std::size_t local = 0;
      while (local < n_named && named[local] != v)
        ++local;
      if (local == n_named)
        named[n_named++] = v;
      ids[s][k] = Index(local);
      const auto &p = points[std::size_t(v)];
      work.corners[3 * s + k] = {Index(local),
                                 tf::exact::pt3<Int>{p[0], p[1], p[2]}};
    }
  const tf::exact::vertex_range<Index, Int> face0 = tf::make_range(
      static_cast<const tf::exact::vertex<Index, Int> *>(work.corners.data()),
      std::size_t(3));
  const tf::exact::vertex_range<Index, Int> face1 =
      tf::make_range(static_cast<const tf::exact::vertex<Index, Int> *>(
                         work.corners.data() + 3),
                     std::size_t(3));
  if (!tf::intersect::exact_boxes_meet(tf::intersect::placed_face_box(face0),
                                       tf::intersect::placed_face_box(face1)))
    return false;
  const auto fp0 = tf::exact::make_face_plane(face0);
  const auto fp1 = tf::exact::make_face_plane(face1);
  if (tf::intersect::certify_shared_feature_pair<Index>(ids[0], ids[1], face0,
                                                        fp0, face1, fp1))
    return false;
  const auto held_by_both = [&ids](Index local) {
    bool in0 = false, in1 = false;
    for (std::size_t k = 0; k < 3; ++k) {
      in0 = in0 || ids[0][k] == local;
      in1 = in1 || ids[1][k] == local;
    }
    return in0 && in1;
  };
  struct face_t {
    Index face;
    const std::array<Index, 3> *corners;
    auto id() const -> Index { return face; }
    auto indices() const -> const std::array<Index, 3> & { return *corners; }
  };
  // the two-face form's membership: a corner both hold lists the second
  // triangle first
  struct membership_t {
    const decltype(held_by_both) *both;
    const std::array<std::array<Index, 3>, 2> *ids;
    struct row_t {
      Index first;
      auto front() const -> Index { return first; }
    };
    auto operator[](Index local) const -> row_t {
      if ((*both)(local))
        return {Index(1)};
      for (std::size_t k = 0; k < 3; ++k)
        if ((*ids)[0][k] == local)
          return {Index(0)};
      return {Index(1)};
    }
  };
  // the two-face form's edge link: an edge both hold is the first
  // triangle's
  struct link_t {
    const decltype(held_by_both) *both;
    const std::array<std::array<Index, 3>, 2> *ids;
    struct edge_t {
      bool shared;
      auto is_representative(Index face) const -> bool {
        return !shared || face == Index(0);
      }
    };
    struct face_edges_t {
      const link_t *link;
      Index face;
      auto operator[](std::size_t k) const -> edge_t {
        const auto &c = (*link->ids)[std::size_t(face)];
        return {(*link->both)(c[k]) && (*link->both)(c[(k + 1) % 3])};
      }
    };
    auto operator[](Index face) const -> face_edges_t { return {this, face}; }
  };
  const membership_t fm{&held_by_both, &ids};
  const link_t mel{&held_by_both, &ids};
  tf::intersect::compute_shared_masks<Index>(ids[0], ids[1], ws.shared0,
                                             ws.shared1);
  tf::intersect::within_polygon_pair_prepped(ws, face_t{Index(0), &ids[0]},
                                             face_t{Index(1), &ids[1]}, 0, mel,
                                             fm, face0, fp0, face1, fp1);
  if (!ws.intersections.size())
    return false;
  // the kernel states no contact at a corner both hold; a pair that meets
  // holds each such corner as a contact of the two, as the form's build
  // delivers it
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      if (ids[0][i] == ids[1][j])
        ws.intersections.push_back({std::int16_t(0),
                                    std::int16_t(0),
                                    Index(0),
                                    Index(1),
                                    {Index(i), tf::topo_type::vertex},
                                    {Index(j), tf::topo_type::vertex},
                                    Index(-1)});
  return true;
}

} // namespace tf::tetmesh
