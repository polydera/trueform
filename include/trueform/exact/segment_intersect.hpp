/*
 * Copyright (c) 2025 XLAB
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

#include "../topology/topo_id.hpp"
#include "./coplanar_edge_edge_point.hpp"
#include "./orient2d.hpp"
#include "./segments_cross.hpp"
#include "./vertex.hpp"

#include <optional>
#include <utility>

namespace tf::exact {

/// Classification of a segment-segment intersection.
/// target_a is w.r.t. segment A, target_b w.r.t. segment B.
/// For a segment (v0, v1):
///   {0, vertex} = at v0
///   {1, vertex} = at v1
///   {0, edge}   = interior
template <typename Index> struct segment_intersection {
  tf::topo_id<Index> target_a;
  tf::topo_id<Index> target_b;
};

// ── point-on-segment ────────────────────────────────────────────────

/// Test if point p lies on segment (a, b) in 2D projection.
/// Returns {0, vertex} if p==a, {1, vertex} if p==b,
///         {0, edge} if interior, nullopt if outside.
inline auto point_on_segment(const vertex &p, const vertex &a, const vertex &b,
                             int ax0, int ax1)
    -> std::optional<tf::topo_id<short>> {
  if (p.pt[0] == a.pt[0] && p.pt[1] == a.pt[1] && p.pt[2] == a.pt[2])
    return tf::topo_id<short>{0, tf::topo_type::vertex};
  if (p.pt[0] == b.pt[0] && p.pt[1] == b.pt[1] && p.pt[2] == b.pt[2])
    return tf::topo_id<short>{1, tf::topo_type::vertex};
  if (orient2d_sign(a, b, p, ax0, ax1) != 0)
    return std::nullopt;
  auto between = [](int32_t lo, int32_t hi, int32_t v) {
    return (lo <= v && v <= hi) || (hi <= v && v <= lo);
  };
  if (between(a.pt[ax0], b.pt[ax0], p.pt[ax0]) &&
      between(a.pt[ax1], b.pt[ax1], p.pt[ax1]))
    return tf::topo_id<short>{0, tf::topo_type::edge};
  return std::nullopt;
}

// ── segments_cross_sos ──────────────────────────────────────────────

/// SoS segment crossing test: 4 orient2d_sos calls, never zero.
inline auto segments_cross_sos(const vertex &a0, const vertex &a1,
                               const vertex &b0, const vertex &b1, int ax0,
                               int ax1) -> bool {
  bool o1 = orient2d_sos(a0, a1, b0, ax0, ax1);
  bool o2 = orient2d_sos(a0, a1, b1, ax0, ax1);
  bool o3 = orient2d_sos(b0, b1, a0, ax0, ax1);
  bool o4 = orient2d_sos(b0, b1, a1, ax0, ax1);
  return (o1 != o2) && (o3 != o4);
}

// ── detection only ──────────────────────────────────────────────────

/// Primitives: full VV/VE/EE classification, no point computation.
/// Topological VV (same point ID = same contour) is skipped.
/// Real VV (different IDs, same coordinates) is reported.
inline auto segment_intersect(const vertex &a0, const vertex &a1,
                              const vertex &b0, const vertex &b1, int ax0,
                              int ax1)
    -> std::optional<segment_intersection<short>> {
  // Topological VV: shared point ID → same contour, skip
  if (a0.id == b0.id || a0.id == b1.id || a1.id == b0.id || a1.id == b1.id)
    return std::nullopt;

  auto pts_equal = [](const pt3 &x, const pt3 &y) {
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
  };

  // Real VV: different IDs, same coordinates → different contours meeting
  if (pts_equal(a0.pt, b0.pt))
    return segment_intersection<short>{
        {0, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (pts_equal(a0.pt, b1.pt))
    return segment_intersection<short>{
        {0, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};
  if (pts_equal(a1.pt, b0.pt))
    return segment_intersection<short>{
        {1, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (pts_equal(a1.pt, b1.pt))
    return segment_intersection<short>{
        {1, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};

  // VE: endpoint of A on interior of B
  if (auto t = point_on_segment(a0, b0, b1, ax0, ax1);
      t && t->label == tf::topo_type::edge)
    return segment_intersection<short>{{0, tf::topo_type::vertex}, *t};
  if (auto t = point_on_segment(a1, b0, b1, ax0, ax1);
      t && t->label == tf::topo_type::edge)
    return segment_intersection<short>{{1, tf::topo_type::vertex}, *t};

  // VE: endpoint of B on interior of A
  if (auto t = point_on_segment(b0, a0, a1, ax0, ax1);
      t && t->label == tf::topo_type::edge)
    return segment_intersection<short>{*t, {0, tf::topo_type::vertex}};
  if (auto t = point_on_segment(b1, a0, a1, ax0, ax1);
      t && t->label == tf::topo_type::edge)
    return segment_intersection<short>{*t, {1, tf::topo_type::vertex}};

  // EE: proper crossing
  pt2 pa0 = {a0.pt[ax0], a0.pt[ax1]}, pa1 = {a1.pt[ax0], a1.pt[ax1]};
  pt2 pb0 = {b0.pt[ax0], b0.pt[ax1]}, pb1 = {b1.pt[ax0], b1.pt[ax1]};
  if (segments_cross(pa0, pa1, pb0, pb1))
    return segment_intersection<short>{
        {0, tf::topo_type::edge}, {0, tf::topo_type::edge}};

  return std::nullopt;
}

/// SoS: real VV by coordinate check, EE by orient2d_sos.
/// Topological VV (same point ID) is skipped.
inline auto segment_intersect_sos(const vertex &a0, const vertex &a1,
                                  const vertex &b0, const vertex &b1, int ax0,
                                  int ax1)
    -> std::optional<segment_intersection<short>> {
  // Topological VV: shared point ID → same contour, skip
  if (a0.id == b0.id || a0.id == b1.id || a1.id == b0.id || a1.id == b1.id)
    return std::nullopt;

  auto pts_equal = [](const pt3 &x, const pt3 &y) {
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
  };

  // Real VV: different IDs, same coordinates → different contours meeting
  if (pts_equal(a0.pt, b0.pt))
    return segment_intersection<short>{
        {0, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (pts_equal(a0.pt, b1.pt))
    return segment_intersection<short>{
        {0, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};
  if (pts_equal(a1.pt, b0.pt))
    return segment_intersection<short>{
        {1, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (pts_equal(a1.pt, b1.pt))
    return segment_intersection<short>{
        {1, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};

  // EE: proper crossing (SoS — no VE possible)
  if (!segments_cross_sos(a0, a1, b0, b1, ax0, ax1))
    return std::nullopt;
  return segment_intersection<short>{
      {0, tf::topo_type::edge}, {0, tf::topo_type::edge}};
}

// ── detection + point ───────────────────────────────────────────────

/// Primitives: full classification + intersection point.
inline auto segment_intersect_point(const vertex &a0, const vertex &a1,
                                    const vertex &b0, const vertex &b1,
                                    int ax0, int ax1)
    -> std::optional<std::pair<segment_intersection<short>, pt3>> {
  auto hit = segment_intersect(a0, a1, b0, b1, ax0, ax1);
  if (!hit)
    return std::nullopt;

  // VV or VE on A side: point is the A vertex
  if (hit->target_a.label == tf::topo_type::vertex)
    return std::pair{*hit, hit->target_a.id == 0 ? a0.pt : a1.pt};

  // VE on B side: point is the B vertex
  if (hit->target_b.label == tf::topo_type::vertex)
    return std::pair{*hit, hit->target_b.id == 0 ? b0.pt : b1.pt};

  // EE: compute crossing point
  return std::pair{*hit, coplanar_edge_edge_point(a0, a1, b0, b1, ax0, ax1)};
}

/// SoS: real VV + proper crossing, with intersection point.
inline auto segment_intersect_point_sos(const vertex &a0, const vertex &a1,
                                        const vertex &b0, const vertex &b1,
                                        int ax0, int ax1)
    -> std::optional<std::pair<segment_intersection<short>, pt3>> {
  auto hit = segment_intersect_sos(a0, a1, b0, b1, ax0, ax1);
  if (!hit)
    return std::nullopt;

  // Real VV: point is the A vertex coordinate
  if (hit->target_a.label == tf::topo_type::vertex)
    return std::pair{*hit, hit->target_a.id == 0 ? a0.pt : a1.pt};

  // EE: compute crossing point
  return std::pair{*hit, coplanar_edge_edge_point(a0, a1, b0, b1, ax0, ax1)};
}

} // namespace tf::exact
