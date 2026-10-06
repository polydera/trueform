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
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../topology/boundary_rims.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_carrier_normal.hpp"
#include "./hole_cycle_is_simple.hpp"
#include "./hole_face_carries_edge.hpp"
#include "./hole_metric.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::fill {

/// What preflight made of one rim: the status it publishes and the rim edge
/// that earned it, `-1` where the rim passed.
template <typename Index> struct hole_preflight_verdict {
  tf::hole_fill_status status;
  Index edge;
};

/// The scratch one rim's preparation walks on, reused across the rims a
/// worker takes.
template <typename Index, typename Int> struct hole_rim_scratch {
  tf::buffer<tf::point<Int, 3>> lattice;
  tf::buffer<Index> order;
  tf::buffer<bool> forward;
  tf::fill::hole_cycle_sweep<Index, Int> sweep;
};

/// Validate one rim, and leave behind what its cycle is stated from.
///
/// The checks run in one order and the first that fails names the status:
/// closed, carrier arity and liveness, vertex distinctness, nonzero edges,
/// single carrier, geometric simplicity, ticket resolution, and last the
/// carrying triangle's geometric usability — which is placed last so that
/// every structural fact is established before a plane is read off a face.
/// Within a check the offending edge is the lowest that fails it.
///
/// A rim that passes leaves its lattice positions and, per rim edge, whether
/// the carrying face winds it the way the rim runs it.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename Converter>
auto validate_hole_rim(const tf::polygons<Policy> &polygons,
                       const tf::face_membership_like<MembershipPolicy> &fm,
                       const tf::boundary_rims<Index> &rims, Index rim,
                       const Converter &converter,
                       tf::fill::hole_rim_scratch<Index, Int> &scratch)
    -> tf::fill::hole_preflight_verdict<Index> {
  using verdict_t = tf::fill::hole_preflight_verdict<Index>;
  const auto vertices = rims.vertices[std::size_t(rim)];
  const auto carriers = rims.faces[std::size_t(rim)];
  const auto mesh_faces = polygons.faces();
  const auto points = polygons.points();
  const Index n = Index(vertices.size());
  const auto next = [n](Index k) { return Index(k + 1 == n ? 0 : k + 1); };

  if (!rims.closed[std::size_t(rim)])
    return verdict_t{tf::hole_fill_status::refused_open, 0};
  if (n < 3)
    return verdict_t{tf::hole_fill_status::refused_invalid_rim, 0};

  for (Index k = 0; k < n; ++k) {
    const Index face = carriers[std::size_t(k)];
    if (face < 0 || std::size_t(face) >= mesh_faces.size() ||
        mesh_faces[std::size_t(face)].size() != 3)
      return verdict_t{tf::hole_fill_status::refused_invalid_rim, k};
  }

  scratch.order.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k)
    scratch.order[std::size_t(k)] = k;
  std::sort(scratch.order.begin(), scratch.order.end(),
            [&vertices](Index a, Index b) {
              const Index va = vertices[std::size_t(a)];
              const Index vb = vertices[std::size_t(b)];
              return va != vb ? va < vb : a < b;
            });
  for (Index k = 1; k < n; ++k)
    if (vertices[std::size_t(scratch.order[std::size_t(k)])] ==
        vertices[std::size_t(scratch.order[std::size_t(k - 1)])])
      return verdict_t{tf::hole_fill_status::refused_invalid_rim,
                       std::min(scratch.order[std::size_t(k)],
                                scratch.order[std::size_t(k - 1)])};

  scratch.lattice.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k)
    scratch.lattice[std::size_t(k)] =
        converter(points[std::size_t(vertices[std::size_t(k)])]);
  for (Index k = 0; k < n; ++k)
    if (scratch.lattice[std::size_t(k)] ==
        scratch.lattice[std::size_t(next(k))])
      return verdict_t{tf::hole_fill_status::refused_invalid_rim, k};

  for (Index k = 0; k < n; ++k) {
    const Index a = vertices[std::size_t(k)];
    const Index b = vertices[std::size_t(next(k))];
    Index count = 0;
    for (auto face : fm[std::size_t(a)])
      count = Index(
          count + (tf::fill::hole_face_carries_edge(mesh_faces, face, a, b)
                       ? 1
                       : 0));
    if (count != 1)
      return verdict_t{tf::hole_fill_status::refused_invalid_rim, k};
  }

  Index offending = Index(-1);
  if (!tf::fill::hole_cycle_is_simple(tf::make_range(scratch.lattice),
                                      scratch.sweep, offending))
    return verdict_t{tf::hole_fill_status::refused_invalid_rim, offending};

  scratch.forward.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k) {
    const Index a = vertices[std::size_t(k)];
    const Index b = vertices[std::size_t(next(k))];
    const auto corners = mesh_faces[std::size_t(carriers[std::size_t(k)])];
    bool wound = false, reversed = false;
    for (Index i = 0; i < 3; ++i) {
      const Index j = Index(i + 1 == 3 ? 0 : i + 1);
      wound = wound || (corners[std::size_t(i)] == a &&
                        corners[std::size_t(j)] == b);
      reversed = reversed || (corners[std::size_t(i)] == b &&
                              corners[std::size_t(j)] == a);
    }
    if (!wound && !reversed)
      return verdict_t{tf::hole_fill_status::refused_invalid_rim, k};
    scratch.forward[std::size_t(k)] = wound;
  }

  for (Index k = 0; k < n; ++k)
    if (!tf::fill::hole_normal_is_usable(tf::fill::hole_carrier_normal(
            mesh_faces, points, carriers[std::size_t(k)])))
      return verdict_t{tf::hole_fill_status::refused_host_face, k};

  return verdict_t{tf::hole_fill_status::filled, Index(-1)};
}

} // namespace tf::fill
