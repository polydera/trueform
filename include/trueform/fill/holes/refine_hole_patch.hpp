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
#include "../../core/angle.hpp"
#include "../../core/buffer.hpp"
#include "../../core/dot.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/triangle_quality.hpp"
#include "../../core/unsafe.hpp"
#include "../../remesh/flip/admits_min_angle_flip.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../../topology/half_edge_handle.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_chord_is_forbidden.hpp"
#include "./hole_metric.hpp"
#include "./hole_patch.hpp"
#include "./split_hole_patch_edge.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::fill {

/// The scratch one patch's refinement walks on.
template <typename Index> struct hole_refine_scratch {
  tf::buffer<double> quality;
  tf::buffer<Index> support;
};

/// The midpoints a refinement may mint per triangle it was handed.
inline constexpr std::size_t hole_refine_mints_per_triangle = 4;
/// The flips a refinement may accept per triangle it was handed.
inline constexpr std::size_t hole_refine_flips_per_triangle = 8;

/// The quality a triangle states, and whether it states one at all.
///
/// A triangle whose doubled area or whose longest side does not evaluate
/// finitely, and one with no longest side, states none — which reads as the
/// worst class for a triangle the patch already holds, and refuses a move
/// that would create it.
inline auto hole_patch_quality(const tf::point<double, 3> &a,
                               const tf::point<double, 3> &b,
                               const tf::point<double, 3> &c, double &quality)
    -> bool {
  const auto normal = tf::fill::hole_oriented_normal(a, b, c);
  const double two_area = std::sqrt(tf::dot(normal, normal));
  const double longest =
      std::max({(b - a).length2(), (c - b).length2(), (a - c).length2()});
  if (!std::isfinite(two_area) || !std::isfinite(longest) || longest <= 0.0)
    return false;
  quality = tf::triangle_quality(two_area, longest);
  return std::isfinite(quality);
}

/// Refine a coarse patch toward a quality floor.
///
/// The worst triangle leads: each round measures the patch, stops the moment
/// the floor holds over all of it, and otherwise offers exactly two moves on
/// the worst triangle — the midpoint of its longest INTERIOR edge, and a flip
/// of one of them. A move is taken only when the smallest quality over its
/// complete support rises by the driver's margin, the supports being every
/// face incident to the vertices the move touches, before and after. A rim
/// edge is never split and never flipped, so the patch keeps its boundary and
/// the host never learns of this tier.
///
/// The budgets are the caller's, frozen before the first move against the
/// triangle count it hands over. One family's exhaustion removes only its own
/// candidates, so the other is still offered; `budget_exhausted` is stated
/// only where a budget is what removed the last improving move, and `stalled`
/// where none remained to remove.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename RealT>
auto refine_hole_patch(const tf::polygons<Policy> &polygons,
                       const tf::face_membership_like<MembershipPolicy> &fm,
                       Index n_points, double floor, std::size_t mint_budget,
                       std::size_t flip_budget,
                       tf::buffer<tf::point<RealT, 3>> &minted,
                       tf::fill::hole_refine_scratch<Index> &scratch,
                       tf::fill::hole_patch<Index, RealT> &patch)
    -> tf::hole_refine_status {
  using handle_t = tf::half_edge_handle<Index>;
  const double delta = 1e-3;
  const auto mesh_faces = polygons.faces();
  std::size_t mints = 0, flips = 0;

  const auto position = [&patch](Index local) {
    return patch.positions[std::size_t(local)].template as<double>();
  };
  const auto stated_quality = [&patch, &position](Index face) {
    const auto c = patch.corners_of(face);
    double value = 0.0;
    return tf::fill::hole_patch_quality(position(c[0]), position(c[1]),
                                        position(c[2]), value)
               ? value
               : 0.0;
  };
  const auto gather_star = [&patch](Index vertex, tf::buffer<Index> &out) {
    const auto start =
        patch.he.vertex_half_edge_handles()[std::size_t(vertex)];
    auto current = start;
    do {
      const Index face = patch.he.half_edge(current).face;
      if (face >= 0 && std::find(out.begin(), out.end(), face) == out.end())
        out.push_back(face);
      current = patch.he.rotated(current);
      if (!current.is_valid())
        break;
    } while (current != start);
  };
  const auto support_minimum = [&scratch](Index left, Index right) {
    double value = std::numeric_limits<double>::infinity();
    for (auto face : scratch.support)
      if (face != left && face != right)
        value = std::min(value, scratch.quality[std::size_t(face)]);
    return value;
  };
  const auto support_before = [&scratch]() {
    double value = std::numeric_limits<double>::infinity();
    for (auto face : scratch.support)
      value = std::min(value, scratch.quality[std::size_t(face)]);
    return value;
  };

  scratch.quality.allocate(patch.n_faces());
  for (std::size_t face = 0; face < patch.n_faces(); ++face)
    scratch.quality[face] = stated_quality(Index(face));

  for (;;) {
    std::size_t worst = 0;
    auto worst_corners = patch.corners_of(0);
    for (std::size_t face = 1; face < patch.n_faces(); ++face) {
      const auto corners = patch.corners_of(Index(face));
      if (scratch.quality[face] < scratch.quality[worst] ||
          (scratch.quality[face] == scratch.quality[worst] &&
           corners < worst_corners)) {
        worst = face;
        worst_corners = corners;
      }
    }
    if (scratch.quality[worst] >= floor) {
      patch.he.rebuild_handles(Index(patch.n_faces()),
                               Index(patch.n_vertices()));
      return tf::hole_refine_status::floor_met;
    }

    const handle_t h0 = patch.he.face_half_edge_handles()[worst];
    const handle_t h1 = patch.he.next(tf::unsafe, h0);
    const handle_t h2 = patch.he.next(tf::unsafe, h1);
    const handle_t sides[3] = {h0, h1, h2};
    const auto edge_key = [&patch](const handle_t &side) {
      const Index a = patch.flat[std::size_t(
          patch.he.start_vertex_handle(tf::unsafe, side).id())];
      const Index b = patch.flat[std::size_t(
          patch.he.end_vertex_handle(tf::unsafe, side).id())];
      return std::array<Index, 2>{std::min(a, b), std::max(a, b)};
    };
    const auto interior = [&patch, &sides](int side) {
      return !patch.he.is_boundary(
          tf::unsafe, patch.he.edge_handle(tf::unsafe, sides[side]));
    };

    double worst_value = 0.0;
    bool states = tf::fill::hole_patch_quality(
        position(worst_corners[0]), position(worst_corners[1]),
        position(worst_corners[2]), worst_value);
    int split_side = -1;
    double split_length = -1.0;
    std::array<Index, 2> split_key{};
    for (int side = 0; side < 3; ++side) {
      if (!interior(side))
        continue;
      const Index a =
          patch.he.start_vertex_handle(tf::unsafe, sides[side]).id();
      const Index b = patch.he.end_vertex_handle(tf::unsafe, sides[side]).id();
      const double length = (position(b) - position(a)).length2();
      states = states && std::isfinite(length);
      const auto key = edge_key(sides[side]);
      if (split_side < 0 || length > split_length ||
          (length == split_length && key < split_key)) {
        split_side = side;
        split_length = length;
        split_key = key;
      }
    }
    if (split_side >= 0 && !states) {
      split_side = -1;
      for (int side = 0; side < 3; ++side) {
        if (!interior(side))
          continue;
        const auto key = edge_key(sides[side]);
        if (split_side < 0 || split_key < key) {
          split_side = side;
          split_key = key;
        }
      }
    }

    bool split_improves = false;
    if (split_side >= 0) {
      const handle_t side = sides[split_side];
      const handle_t mate = patch.he.opposite(tf::unsafe, side);
      const Index u = patch.he.start_vertex_handle(tf::unsafe, side).id();
      const Index v = patch.he.end_vertex_handle(tf::unsafe, side).id();
      const Index w = patch.he
                          .end_vertex_handle(tf::unsafe,
                                             patch.he.next(tf::unsafe, side))
                          .id();
      const Index x = patch.he
                          .end_vertex_handle(tf::unsafe,
                                             patch.he.next(tf::unsafe, mate))
                          .id();
      const auto pu = position(u), pv = position(v);
      const tf::point<double, 3> pm{0.5 * (pu[0] + pv[0]),
                                    0.5 * (pu[1] + pv[1]),
                                    0.5 * (pu[2] + pv[2])};
      double quarters[4] = {0.0, 0.0, 0.0, 0.0};
      if (tf::fill::hole_patch_quality(pu, pm, position(w), quarters[0]) &&
          tf::fill::hole_patch_quality(pm, pv, position(w), quarters[1]) &&
          tf::fill::hole_patch_quality(pv, pm, position(x), quarters[2]) &&
          tf::fill::hole_patch_quality(pm, pu, position(x), quarters[3])) {
        scratch.support.clear();
        gather_star(u, scratch.support);
        gather_star(v, scratch.support);
        double after = support_minimum(patch.he.half_edge(side).face,
                                       patch.he.half_edge(mate).face);
        for (auto value : quarters)
          after = std::min(after, value);
        split_improves = after > support_before() + delta;
      }
    }

    int flip_side = -1;
    double flip_after = 0.0;
    std::array<Index, 2> flip_key{};
    for (int side = 0; side < 3; ++side) {
      if (!interior(side))
        continue;
      const auto eh = patch.he.edge_handle(tf::unsafe, sides[side]);
      if (!tf::remesh::admits_min_angle_flip(
              patch.he, patch.positions.points(), tf::none, eh,
              tf::rad<RealT>(RealT(0)), RealT(0)))
        continue;
      const handle_t a0 = patch.he.half_edge_handle(tf::unsafe, eh, false);
      const handle_t b0 = patch.he.half_edge_handle(tf::unsafe, eh, true);
      const Index u = patch.he.start_vertex_handle(tf::unsafe, a0).id();
      const Index v = patch.he.start_vertex_handle(tf::unsafe, b0).id();
      const Index w =
          patch.he.end_vertex_handle(tf::unsafe, patch.he.next(tf::unsafe, a0))
              .id();
      const Index x =
          patch.he.end_vertex_handle(tf::unsafe, patch.he.next(tf::unsafe, b0))
              .id();
      if (tf::fill::hole_chord_is_forbidden(
              mesh_faces, fm, n_points, patch.flat[std::size_t(w)],
              patch.carrier[std::size_t(w)], patch.flat[std::size_t(x)],
              patch.carrier[std::size_t(x)]))
        continue;
      double halves[2] = {0.0, 0.0};
      if (!tf::fill::hole_patch_quality(position(w), position(x), position(v),
                                        halves[0]) ||
          !tf::fill::hole_patch_quality(position(x), position(w), position(u),
                                        halves[1]))
        continue;
      scratch.support.clear();
      gather_star(u, scratch.support);
      gather_star(v, scratch.support);
      gather_star(w, scratch.support);
      gather_star(x, scratch.support);
      const double after =
          std::min(support_minimum(patch.he.half_edge(a0).face,
                                   patch.he.half_edge(b0).face),
                   std::min(halves[0], halves[1]));
      if (after <= support_before() + delta)
        continue;
      const auto key = edge_key(sides[side]);
      if (flip_side < 0 || after > flip_after ||
          (after == flip_after && key < flip_key)) {
        flip_side = side;
        flip_after = after;
        flip_key = key;
      }
    }

    bool budget_removed = false;
    if (split_improves && mints >= mint_budget) {
      split_improves = false;
      budget_removed = true;
    }
    if (flip_side >= 0 && flips >= flip_budget) {
      flip_side = -1;
      budget_removed = true;
    }
    if (!split_improves && flip_side < 0) {
      patch.he.rebuild_handles(Index(patch.n_faces()),
                               Index(patch.n_vertices()));
      return budget_removed ? tf::hole_refine_status::budget_exhausted
                            : tf::hole_refine_status::stalled;
    }

    if (split_improves) {
      const handle_t side = sides[split_side];
      const auto eh = patch.he.edge_handle(tf::unsafe, side);
      const handle_t mate = patch.he.opposite(tf::unsafe, side);
      const Index u = patch.he.start_vertex_handle(tf::unsafe, side).id();
      const Index v = patch.he.end_vertex_handle(tf::unsafe, side).id();
      const Index left = patch.he.half_edge(side).face;
      const Index right = patch.he.half_edge(mate).face;
      const tf::point<RealT, 3> mid{
          (patch.positions[std::size_t(u)][0] +
           patch.positions[std::size_t(v)][0]) /
              RealT(2),
          (patch.positions[std::size_t(u)][1] +
           patch.positions[std::size_t(v)][1]) /
              RealT(2),
          (patch.positions[std::size_t(u)][2] +
           patch.positions[std::size_t(v)][2]) /
              RealT(2)};
      const Index m = Index(patch.n_vertices());
      patch.positions.push_back(mid);
      patch.flat.push_back(Index(n_points + Index(minted.size())));
      patch.carrier.push_back(Index(-1));
      minted.push_back(mid);
      const Index born = Index(patch.n_faces());
      tf::fill::split_hole_patch_edge(patch.he, eh, m);
      scratch.quality[std::size_t(left)] = stated_quality(left);
      scratch.quality[std::size_t(right)] = stated_quality(right);
      scratch.quality.push_back(stated_quality(born));
      scratch.quality.push_back(stated_quality(Index(born + 1)));
      ++mints;
      continue;
    }

    const auto eh = patch.he.edge_handle(tf::unsafe, sides[flip_side]);
    const Index left =
        patch.he.half_edge(patch.he.half_edge_handle(tf::unsafe, eh, false))
            .face;
    const Index right =
        patch.he.half_edge(patch.he.half_edge_handle(tf::unsafe, eh, true))
            .face;
    patch.he.flip(eh);
    scratch.quality[std::size_t(left)] = stated_quality(left);
    scratch.quality[std::size_t(right)] = stated_quality(right);
    ++flips;
  }
}

} // namespace tf::fill
