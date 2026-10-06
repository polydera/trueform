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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "./hole_metric.hpp"
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::fill {

/// Stage one over the bands: the least largest charge each state can reach,
/// and whether it can reach one at all. A state of span one is a rim edge and
/// charges nothing.
template <typename Index, typename Int, typename RealT>
auto state_hole_table_bottlenecks(const tf::fill::hole_rim<Index, Int, RealT>
                                      &rim,
                                  Index n, tf::fill::hole_table<Index> &table)
    -> void {
  const auto infinity = std::numeric_limits<double>::infinity();
  const std::size_t root = 2 * table.facets.size();

  for (std::size_t span = 1; span < std::size_t(n); ++span) {
    const auto band = table.bands[span];
    if (span == 1) {
      for (auto state : band) {
        table.bottleneck[std::size_t(state)] = 0.0;
        table.valid[std::size_t(state)] = 1;
        table.choice[std::size_t(state)] = Index(-1);
      }
      continue;
    }
    tf::parallel_for_each(
        band,
        [&rim, &table, n, root, infinity](Index state) {
          const auto outer = std::size_t(state) == root
                                 ? rim.edges[std::size_t(n - 1)].normal
                                 : table.normal[std::size_t(state) / 2];
          double best = infinity;
          char admitted = 0;
          const Index block = table.block[std::size_t(state)];
          if (block >= 0)
            for (const auto &row : table.candidates[std::size_t(block)]) {
              const auto facet = std::size_t(row[3]);
              const std::size_t left = 2 * facet, right = 2 * facet + 1;
              if (!table.valid[left] || !table.valid[right])
                continue;
              const double charge = std::max(
                  table.boundary[facet],
                  tf::fill::hole_dihedral_angle(table.normal[facet], outer));
              if (!std::isfinite(charge))
                continue;
              const double value = std::max(
                  charge,
                  std::max(table.bottleneck[left], table.bottleneck[right]));
              if (!admitted || value < best) {
                best = value;
                admitted = 1;
              }
            }
          table.bottleneck[std::size_t(state)] = best;
          table.valid[std::size_t(state)] = admitted;
          table.choice[std::size_t(state)] = Index(-1);
        },
        tf::checked(tf::fill::hole_table_band_grain));
  }
}

} // namespace tf::fill
