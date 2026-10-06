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

/// Stage two over the same bands: the least area each state can cover under
/// the frozen threshold, and the candidate that covers it. A candidate whose
/// children cover nothing is refused before its angle is ever taken.
template <typename Index, typename Int, typename RealT>
auto state_hole_table_areas(const tf::fill::hole_rim<Index, Int, RealT> &rim,
                            Index n, double threshold,
                            tf::fill::hole_table<Index> &table) -> void {
  const auto infinity = std::numeric_limits<double>::infinity();
  const std::size_t root = 2 * table.facets.size();

  for (std::size_t span = 1; span < std::size_t(n); ++span) {
    const auto band = table.bands[span];
    if (span == 1) {
      for (auto state : band) {
        table.total[std::size_t(state)] = 0.0;
        table.choice[std::size_t(state)] = Index(-1);
      }
      continue;
    }
    tf::parallel_for_each(
        band,
        [&rim, &table, n, root, threshold, infinity](Index state) {
          const auto outer = std::size_t(state) == root
                                 ? rim.edges[std::size_t(n - 1)].normal
                                 : table.normal[std::size_t(state) / 2];
          double best = infinity;
          Index pick = Index(-1);
          const Index block = table.block[std::size_t(state)];
          if (block >= 0)
            for (const auto &row : table.candidates[std::size_t(block)]) {
              const auto facet = std::size_t(row[3]);
              const std::size_t left = 2 * facet, right = 2 * facet + 1;
              if (table.total[left] == infinity ||
                  table.total[right] == infinity)
                continue;
              const double charge = std::max(
                  table.boundary[facet],
                  tf::fill::hole_dihedral_angle(table.normal[facet], outer));
              if (!std::isfinite(charge) || charge > threshold)
                continue;
              const double value =
                  table.area[facet] + table.total[left] + table.total[right];
              if (value < best) {
                best = value;
                pick = Index(facet);
              }
            }
          table.total[std::size_t(state)] = best;
          table.choice[std::size_t(state)] = pick;
        },
        tf::checked(tf::fill::hole_table_band_grain));
  }
}

} // namespace tf::fill
