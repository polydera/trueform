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
#include "../../core/points.hpp"
#include "../hole_split.hpp"
#include "./hole_rim.hpp"
#include "./hole_split_position.hpp"
#include "./sample_hole_chart.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::fill {

/// A chart's extended rim named in the patch's flat identity space: per rim
/// position its flat identity and the carrying triangle a split inherits,
/// `-1` for an original vertex; then the splits in their naming order, each
/// with its record and its point.
template <typename Index, typename RealT> struct hole_chart_rim_names {
  tf::buffer<Index> flat;
  tf::buffer<Index> carrier;
  tf::buffer<tf::hole_split<Index>> records;
  tf::buffer<tf::point<RealT, 3>> points;
  tf::buffer<Index> order;
  tf::buffer<Index> rank;
};

/// Name a chart's extended rim: an original position is its own vertex, and
/// the splits are minted first, `n_points + r` in the canonical order of
/// their records — the original edge's canonical key, then the parameter
/// along it — exactly as the table tier names its own, so a mint's identity
/// is fixed before any sample is named after it.
template <typename PointsPolicy, typename Index, typename Int, typename RealT>
auto name_hole_chart_rim(
    const tf::points<PointsPolicy> &points,
    const tf::fill::hole_rim<Index, Int, RealT> &rim, Index n_points,
    const tf::buffer<tf::fill::hole_chart_rim_split<Index, Int>> &splits,
    tf::fill::hole_chart_rim_names<Index, RealT> &names) -> void {
  const std::size_t n = rim.size();
  const std::size_t s = splits.size();
  const auto parameter_of = [&rim, &splits](std::size_t j) {
    const auto &edge = rim.edges[std::size_t(splits[j].edge)];
    return std::uint8_t(edge.t0 == 0
                            ? int(splits[j].along)
                            : tf::hole_split_scale - int(splits[j].along));
  };

  names.order.clear();
  for (std::size_t j = 0; j < s; ++j)
    names.order.push_back(Index(j));
  std::sort(
      names.order.begin(), names.order.end(),
      [&rim, &splits, &parameter_of](Index x, Index y) {
        const auto &a = rim.edges[std::size_t(splits[std::size_t(x)].edge)];
        const auto &b = rim.edges[std::size_t(splits[std::size_t(y)].edge)];
        if (a.v0 != b.v0)
          return a.v0 < b.v0;
        if (a.v1 != b.v1)
          return a.v1 < b.v1;
        return parameter_of(std::size_t(x)) < parameter_of(std::size_t(y));
      });
  names.rank.allocate(s);
  names.records.clear();
  names.points.clear();
  for (std::size_t r = 0; r < s; ++r) {
    const std::size_t j = std::size_t(names.order[r]);
    const auto &edge = rim.edges[std::size_t(splits[j].edge)];
    const auto parameter = parameter_of(j);
    names.records.push_back(
        {edge.face, edge.v0, edge.v1, Index(n_points + Index(r)), parameter});
    names.points.push_back(tf::fill::hole_split_position(
        points, edge.v0, edge.v1, int(parameter)));
    names.rank[j] = Index(r);
  }

  names.flat.clear();
  names.carrier.clear();
  std::size_t at = 0;
  for (std::size_t k = 0; k < n; ++k) {
    names.flat.push_back(rim.corners[k]);
    names.carrier.push_back(Index(-1));
    for (; at < s && std::size_t(splits[at].edge) == k; ++at) {
      names.flat.push_back(Index(n_points + names.rank[at]));
      names.carrier.push_back(rim.edges[k].face);
    }
  }
}

} // namespace tf::fill
