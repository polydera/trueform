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
template <typename Index, typename Int>
auto point_on_segment(const vertex<Index, Int> &p,
                      const vertex<Index, Int> &a,
                      const vertex<Index, Int> &b, int ax0, int ax1)
    -> std::optional<tf::topo_id<short>> {
  if (p.pt[0] == a.pt[0] && p.pt[1] == a.pt[1] && p.pt[2] == a.pt[2])
    return tf::topo_id<short>{0, tf::topo_type::vertex};
  if (p.pt[0] == b.pt[0] && p.pt[1] == b.pt[1] && p.pt[2] == b.pt[2])
    return tf::topo_id<short>{1, tf::topo_type::vertex};
  if (orient2d_sign(a, b, p, ax0, ax1) != 0)
    return std::nullopt;
  auto between = [](Int lo, Int hi, Int v) {
    return (lo <= v && v <= hi) || (hi <= v && v <= lo);
  };
  if (between(a.pt[ax0], b.pt[ax0], p.pt[ax0]) &&
      between(a.pt[ax1], b.pt[ax1], p.pt[ax1]))
    return tf::topo_id<short>{0, tf::topo_type::edge};
  return std::nullopt;
}

// ── segments_cross_sos ──────────────────────────────────────────────

/// SoS segment crossing test: 4 orient2d_sos calls, never zero.
template <typename Index, typename Int>
auto segments_cross_sos(const vertex<Index, Int> &a0,
                        const vertex<Index, Int> &a1,
                        const vertex<Index, Int> &b0,
                        const vertex<Index, Int> &b1, int ax0, int ax1)
    -> bool {
  bool o1 = orient2d_sos(a0, a1, b0, ax0, ax1);
  bool o2 = orient2d_sos(a0, a1, b1, ax0, ax1);
  bool o3 = orient2d_sos(b0, b1, a0, ax0, ax1);
  bool o4 = orient2d_sos(b0, b1, a1, ax0, ax1);
  return (o1 != o2) && (o3 != o4);
}

// ── full classification ─────────────────────────────────────────────

/// Full classification of two segments. Returns up to 2 contacts.
///
/// Handles: coordinate VV (skips topological VV with shared IDs),
/// VE including containment (both endpoints of one on the other),
/// and EE proper crossing.
///
/// Returns:
///   nullopt                 — no contact
///   {hit, nullopt}          — one contact (VV, VE, or EE)
///   {hit1, hit2}            — two contacts (containment or overlap)
template <typename Index, typename Int>
auto classify_segments(const vertex<Index, Int> &a0,
                       const vertex<Index, Int> &a1,
                       const vertex<Index, Int> &b0,
                       const vertex<Index, Int> &b1, int ax0, int ax1)
    -> std::optional<std::pair<segment_intersection<short>,
                               std::optional<segment_intersection<short>>>> {
  auto pts_equal = [](const pt3<Int> &x, const pt3<Int> &y) {
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
  };

  segment_intersection<short> buf[4];
  int n = 0;

  // VV: coordinate equality, skip topological (same ID)
  if (a0.id != b0.id && pts_equal(a0.pt, b0.pt))
    buf[n++] = {{0, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (a0.id != b1.id && pts_equal(a0.pt, b1.pt))
    buf[n++] = {{0, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};
  if (a1.id != b0.id && pts_equal(a1.pt, b0.pt))
    buf[n++] = {{1, tf::topo_type::vertex}, {0, tf::topo_type::vertex}};
  if (a1.id != b1.id && pts_equal(a1.pt, b1.pt))
    buf[n++] = {{1, tf::topo_type::vertex}, {1, tf::topo_type::vertex}};

  if (n >= 2)
    return std::pair{buf[0], std::optional{buf[1]}};

  // VE: endpoint of A on interior of B
  if (a0.id != b0.id && a0.id != b1.id)
    if (auto t = point_on_segment(a0, b0, b1, ax0, ax1);
        t && t->label == tf::topo_type::edge)
      buf[n++] = {{0, tf::topo_type::vertex}, *t};
  if (a1.id != b0.id && a1.id != b1.id)
    if (auto t = point_on_segment(a1, b0, b1, ax0, ax1);
        t && t->label == tf::topo_type::edge)
      buf[n++] = {{1, tf::topo_type::vertex}, *t};

  if (n >= 2)
    return std::pair{buf[0], std::optional{buf[1]}};

  // VE: endpoint of B on interior of A
  if (b0.id != a0.id && b0.id != a1.id)
    if (auto t = point_on_segment(b0, a0, a1, ax0, ax1);
        t && t->label == tf::topo_type::edge)
      buf[n++] = {*t, {0, tf::topo_type::vertex}};
  if (b1.id != a0.id && b1.id != a1.id)
    if (auto t = point_on_segment(b1, a0, a1, ax0, ax1);
        t && t->label == tf::topo_type::edge)
      buf[n++] = {*t, {1, tf::topo_type::vertex}};

  if (n >= 2)
    return std::pair{buf[0], std::optional{buf[1]}};
  if (n == 1)
    return std::pair{buf[0], std::optional<segment_intersection<short>>{}};

  // EE: proper crossing (only if no VV/VE)
  pt2<Int> pa0 = {a0.pt[ax0], a0.pt[ax1]}, pa1 = {a1.pt[ax0], a1.pt[ax1]};
  pt2<Int> pb0 = {b0.pt[ax0], b0.pt[ax1]}, pb1 = {b1.pt[ax0], b1.pt[ax1]};
  if (segments_cross(pa0, pa1, pb0, pb1))
    return std::pair{
        segment_intersection<short>{{0, tf::topo_type::edge},
                                    {0, tf::topo_type::edge}},
        std::optional<segment_intersection<short>>{}};

  return std::nullopt;
}

// ── detection only ──────────────────────────────────────────────────

/// Primitives: full VV/VE/EE classification, no point computation.
/// Topological VV (same point ID = same contour) is skipped.
/// Real VV (different IDs, same coordinates) is reported.
/// @note Superseded by classify_segments for crossing detection.
///       Kept for segment_intersect_point and base_loops.
template <typename Index, typename Int>
auto segment_intersect(const vertex<Index, Int> &a0,
                       const vertex<Index, Int> &a1,
                       const vertex<Index, Int> &b0,
                       const vertex<Index, Int> &b1, int ax0, int ax1)
    -> std::optional<segment_intersection<short>> {
  // Topological VV: shared point ID → same contour, skip
  if (a0.id == b0.id || a0.id == b1.id || a1.id == b0.id || a1.id == b1.id)
    return std::nullopt;

  auto pts_equal = [](const pt3<Int> &x, const pt3<Int> &y) {
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
  pt2<Int> pa0 = {a0.pt[ax0], a0.pt[ax1]}, pa1 = {a1.pt[ax0], a1.pt[ax1]};
  pt2<Int> pb0 = {b0.pt[ax0], b0.pt[ax1]}, pb1 = {b1.pt[ax0], b1.pt[ax1]};
  if (segments_cross(pa0, pa1, pb0, pb1))
    return segment_intersection<short>{
        {0, tf::topo_type::edge}, {0, tf::topo_type::edge}};

  return std::nullopt;
}

/// SoS: real VV by coordinate check, EE by orient2d_sos.
/// Topological VV (same point ID) is skipped.
template <typename Index, typename Int>
auto segment_intersect_sos(const vertex<Index, Int> &a0,
                           const vertex<Index, Int> &a1,
                           const vertex<Index, Int> &b0,
                           const vertex<Index, Int> &b1, int ax0, int ax1)
    -> std::optional<segment_intersection<short>> {
  // Topological VV: shared point ID → same contour, skip
  if (a0.id == b0.id || a0.id == b1.id || a1.id == b0.id || a1.id == b1.id)
    return std::nullopt;

  auto pts_equal = [](const pt3<Int> &x, const pt3<Int> &y) {
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
template <typename Index, typename Int>
auto segment_intersect_point(const vertex<Index, Int> &a0,
                             const vertex<Index, Int> &a1,
                             const vertex<Index, Int> &b0,
                             const vertex<Index, Int> &b1, int ax0, int ax1)
    -> std::optional<std::pair<segment_intersection<short>, pt3<Int>>> {
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
template <typename Index, typename Int>
auto segment_intersect_point_sos(const vertex<Index, Int> &a0,
                                 const vertex<Index, Int> &a1,
                                 const vertex<Index, Int> &b0,
                                 const vertex<Index, Int> &b1, int ax0,
                                 int ax1)
    -> std::optional<std::pair<segment_intersection<short>, pt3<Int>>> {
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
