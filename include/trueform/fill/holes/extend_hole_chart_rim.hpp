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
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../hole_split.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include "./hole_split_position.hpp"
#include "./sample_hole_chart.hpp"
#include "./subdivide_carrier_triangle.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

/// The scratch one rim's extension walks on.
template <typename Index, typename Int> struct hole_chart_extend_scratch {
  tf::buffer<Index> extended;
  tf::buffer<Index> placed;
  tf::buffer<Index> first;
  tf::buffer<Index> vertices;
  tf::buffer<tf::point<double, 2>> uv;
  tf::buffer<tf::point<Int, 2>> lattice;
  tf::buffer<double> heights;
  tf::buffer<std::array<Index, 3>> ring_triangles;
  tf::buffer<tf::hole_split<Index>> records;
  tf::fill::hole_plan_scratch<Index> plan;
};

/// Grow a chart's rim by the splits its sampling made, the rim given as host
/// vertex ids in its own order.
///
/// The collar's rim block becomes the extended rim of
/// @ref tf::fill::sample_hole_chart — a split position names no host vertex
/// and holds `-1` — and the rings follow it unchanged. A split stands where
/// its record names it, on the original edge
/// (@ref tf::fill::hole_split_position), so its height above the chart is
/// that point's. A ring-1 face whose rim edge a split cut is stated as its
/// canonical sequential subdivision, the plan the stitched mesh will hold, so
/// the ring triangles still cover the host's own faces; `faces` keeps naming
/// the host faces themselves.
template <typename Policy, typename RimRange, typename Index, typename Int>
auto extend_hole_chart_rim(
    const tf::polygons<Policy> &polygons, const RimRange &rim,
    const tf::buffer<tf::fill::hole_chart_rim_split<Index, Int>> &splits,
    tf::fill::hole_chart_extend_scratch<Index, Int> &scratch,
    tf::fill::hole_collar<Index> &collar, tf::fill::hole_chart<Int> &chart)
    -> void {
  const std::size_t s = splits.size();
  if (!s)
    return;
  const auto points = polygons.points();
  const std::size_t n = collar.n_rim();
  const std::size_t total = collar.vertices.size();
  const auto next = [n](std::size_t k) { return k + 1 == n ? 0 : k + 1; };

  scratch.extended.allocate(n);
  scratch.placed.allocate(s);
  scratch.first.allocate(n + 1);
  scratch.vertices.clear();
  scratch.uv.clear();
  scratch.lattice.clear();
  scratch.heights.clear();
  std::size_t at = 0;
  for (std::size_t k = 0; k < n; ++k) {
    scratch.first[k] = Index(at);
    scratch.extended[k] = Index(scratch.vertices.size());
    scratch.vertices.push_back(collar.vertices[k]);
    scratch.uv.push_back(chart.uv[k]);
    scratch.lattice.push_back(chart.lattice[k]);
    scratch.heights.push_back(chart.heights[k]);
    for (; at < s && std::size_t(splits[at].edge) == k; ++at) {
      const Index a = Index(rim[k]), b = Index(rim[next(k)]);
      const bool forward = a < b;
      const auto position = tf::fill::hole_split_position(
          points, forward ? a : b, forward ? b : a,
          forward ? int(splits[at].along)
                  : tf::hole_split_scale - int(splits[at].along));
      scratch.placed[at] = Index(scratch.vertices.size());
      scratch.vertices.push_back(Index(-1));
      scratch.uv.push_back(splits[at].uv);
      scratch.lattice.push_back(splits[at].lattice);
      scratch.heights.push_back(
          tf::dot(position.template as<double>() - chart.origin, chart.normal));
    }
  }
  scratch.first[n] = Index(s);
  for (std::size_t k = n; k < total; ++k) {
    scratch.vertices.push_back(collar.vertices[k]);
    scratch.uv.push_back(chart.uv[k]);
    scratch.lattice.push_back(chart.lattice[k]);
    scratch.heights.push_back(chart.heights[k]);
  }

  const auto lifted = [&scratch, n, s](Index position) {
    return std::size_t(position) < n ? scratch.extended[std::size_t(position)]
                                     : Index(std::size_t(position) + s);
  };
  scratch.ring_triangles.clear();
  for (const auto &triangle : collar.ring_triangles) {
    scratch.records.clear();
    for (std::size_t side = 0; side < 3; ++side) {
      const std::size_t p = std::size_t(triangle[side]);
      const std::size_t q = std::size_t(triangle[(side + 1) % 3]);
      if (p >= n || q >= n)
        continue;
      const bool forward = q == next(p);
      if (!forward && p != next(q))
        continue;
      const std::size_t edge = forward ? p : q;
      const Index from = scratch.extended[edge];
      const Index to = scratch.extended[next(edge)];
      for (Index j = scratch.first[edge]; j < scratch.first[edge + 1]; ++j)
        scratch.records.push_back({Index(-1), from, to,
                                   scratch.placed[std::size_t(j)],
                                   splits[std::size_t(j)].along});
    }
    const std::array<Index, 3> corners{lifted(triangle[0]), lifted(triangle[1]),
                                       lifted(triangle[2])};
    if (!scratch.records.size()) {
      scratch.ring_triangles.push_back(corners);
      continue;
    }
    tf::fill::subdivide_carrier_triangle(corners,
                                         tf::make_range(scratch.records),
                                         scratch.plan, scratch.ring_triangles);
  }

  collar.vertices.clear();
  for (auto vertex : scratch.vertices)
    collar.vertices.push_back(vertex);
  collar.vertex_offsets = {Index(0), Index(n + s),
                           Index(std::size_t(collar.vertex_offsets[2]) + s),
                           Index(std::size_t(collar.vertex_offsets[3]) + s)};
  collar.ring_triangles.clear();
  for (const auto &triangle : scratch.ring_triangles)
    collar.ring_triangles.push_back(triangle);
  chart.uv.allocate(scratch.uv.size());
  chart.lattice.allocate(scratch.lattice.size());
  chart.heights.allocate(scratch.heights.size());
  for (std::size_t k = 0; k < scratch.uv.size(); ++k) {
    chart.uv[k] = scratch.uv[k];
    chart.lattice[k] = scratch.lattice[k];
    chart.heights[k] = scratch.heights[k];
  }
}

} // namespace tf::fill
