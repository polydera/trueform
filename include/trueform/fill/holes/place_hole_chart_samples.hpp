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
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./hole_chart.hpp"
#include "./hole_trend.hpp"
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Stand every sample at the trend's height plus its residual, in the mesh's
/// own coordinate type.
///
/// Returns false where the trend does not cover a sample, or a position does
/// not evaluate finitely; `out` then holds nothing.
template <typename Int, typename Samples, typename RealT>
auto place_hole_chart_samples(const tf::fill::hole_chart<Int> &chart,
                              const tf::fill::hole_trend &trend,
                              const Samples &samples,
                              const tf::buffer<double> &residual,
                              tf::buffer<tf::point<RealT, 3>> &out) -> bool {
  const std::size_t count = samples.size();
  out.allocate(count);
  bool placed_all = true;
  tf::blocked_reduce(
      tf::make_sequence_range(count), placed_all,
      [&](const auto &block, bool &local) {
        for (std::size_t j : block) {
          const tf::point<double, 2> at{samples[j][0], samples[j][1]};
          double height = 0.0;
          if (!tf::fill::hole_trend_height(trend, chart, at, height)) {
            local = false;
            return;
          }
          const auto placed =
              chart.lift(at, height + residual[j]).template as<RealT>();
          for (std::size_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(placed[axis])) {
              local = false;
              return;
            }
          out[j] = placed;
        }
      },
      [](bool local, bool &all) { all = all && local; }, tf::checked);
  if (!placed_all) {
    out.clear();
    return false;
  }
  return true;
}

} // namespace tf::fill
