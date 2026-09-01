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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"

#include <cmath>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Side of a sheet for a batch of query points, by generalized
///        winding number.
///
/// One parallel pass over the sheet's triangles accumulates the solid
/// angle of every triangle at every query simultaneously (vertices
/// fetched once, amortised over the batch). Winding has no feature
/// cases where pseudo-normal classification fails. Brute force on
/// purpose: a tree-accelerated evaluator can replace the body behind
/// the same signature, with this as its oracle.
///
/// @param form    The sheet form (faces + points, frame respected).
/// @param queries Query points in exact pipeline coordinates.
/// @param read_point `(vertex id) -> pt3<Int>` — where the sheet's own
///        vertices stand, which is the placed position once a door ran.
/// @return One char per query: 1 behind the sheet's normal, else 0.
template <typename Form, typename Int, typename ReadPoint>
auto winding_side(const Form &form,
                  const tf::buffer<tf::exact::pt3<Int>> &queries,
                  const ReadPoint &read_point) -> tf::buffer<char> {
  const std::size_t m = queries.size();
  tf::buffer<char> side;
  side.allocate(m);
  if (m == 0)
    return side;

  tf::buffer<double> total;
  total.allocate(m);
  for (std::size_t j = 0; j < m; ++j)
    total[j] = 0.0;

  auto exact_point = [&](auto id) -> tf::exact::pt3<Int> {
    return read_point(id);
  };

  struct local_t {
    tf::buffer<double> acc;
  };

  auto task = [&](auto &&faces, local_t &local) {
    if (local.acc.size() != m) {
      local.acc.allocate(m);
      for (std::size_t j = 0; j < m; ++j)
        local.acc[j] = 0.0;
    }
    using T1 = typename tf::exact::meta<Int>::T1;
    for (auto &&face : faces) {
      const auto p0 = exact_point(face[0]);
      const auto p1 = exact_point(face[1]);
      const auto p2 = exact_point(face[2]);
      for (std::size_t j = 0; j < m; ++j) {
        const auto &q = queries[j];
        // Differences widened to T1 (full-range Int would overflow),
        // exact, then double.
        const double ax = double(T1(p0[0]) - T1(q[0])),
                     ay = double(T1(p0[1]) - T1(q[1])),
                     az = double(T1(p0[2]) - T1(q[2]));
        const double bx = double(T1(p1[0]) - T1(q[0])),
                     by = double(T1(p1[1]) - T1(q[1])),
                     bz = double(T1(p1[2]) - T1(q[2]));
        const double cx = double(T1(p2[0]) - T1(q[0])),
                     cy = double(T1(p2[1]) - T1(q[1])),
                     cz = double(T1(p2[2]) - T1(q[2]));
        const double la = tf::sqrt(ax * ax + ay * ay + az * az);
        const double lb = tf::sqrt(bx * bx + by * by + bz * bz);
        const double lc = tf::sqrt(cx * cx + cy * cy + cz * cz);
        const double det = ax * (by * cz - bz * cy) -
                           ay * (bx * cz - bz * cx) +
                           az * (bx * cy - by * cx);
        const double den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc +
                           (bx * cx + by * cy + bz * cz) * la +
                           (cx * ax + cy * ay + cz * az) * lb;
        local.acc[j] += 2.0 * std::atan2(det, den);
      }
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    if (local.acc.size() != m)
      return;
    for (std::size_t j = 0; j < m; ++j)
      total[j] += local.acc[j];
  };

  tf::blocked_reduce(form.faces(), tf::none, local_t{},
                                         task, agg);

  // Threshold at w = 1/4 (total = pi): the midpoint between the
  // negative-side/outside values {-2*pi, 0} and the behind/inside
  // values {+2*pi, +4*pi}, so open sheets and closed surfaces declared
  // as sheets both classify with maximal margin. In particular a
  // closed sheet's exterior totals exactly 0 — a sign test would be a
  // coin flip there.
  // Sign test with a dead zone just above the accumulation noise
  // floor: a genuine back side is at least the sheet's subtended
  // angle (a bounded sheet seen from afar or near its rim can be far
  // below 2*pi, so no fixed fraction of the winding works), while a
  // closed sheet's exterior totals exactly 0 plus float noise.
  constexpr double noise_floor = 1e-6;
  for (std::size_t j = 0; j < m; ++j)
    side[j] = total[j] > noise_floor ? char(1) : char(0);
  return side;
}

} // namespace tf::csg::graph
