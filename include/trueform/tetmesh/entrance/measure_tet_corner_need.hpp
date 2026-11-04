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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/checked.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/mapped_range.hpp"
#include "./tet_corner_need.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace tf::tetmesh {

/// The frame depth the corners of `polygons` ask and the narrowest of
/// them. A shell of radius `r` about a corner of angle `t` holds its pieces
/// equal to `t^2 / 2` of `r`, and a frame of `b` bits places a split on a
/// piece of length `l` to `2^-b l`, so a corner asks
///
///   b >= log2(l / r) + log2(2 / t^2)
///
/// with `l` the longer and `2 r` the shorter of the corner's two edges;
/// `corner_bits` is the most any corner asks. The polygons may be the
/// input's or the entrance's, whose cuts make corners of their own.
template <typename Policy>
auto measure_tet_corner_need(const tf::polygons<Policy> &polygons)
    -> tet_corner_need {
  using limits = std::numeric_limits<tf::coordinate_type<Policy>>;
  const int ceiling = 4 * limits::max_exponent;
  const auto corners = tf::reduce(
      tf::make_mapped_range(
          polygons,
          [](const auto &polygon) {
            std::pair<double, double> worst{0, 180};
            const std::size_t n = polygon.size();
            for (std::size_t i = 0; i < n; ++i) {
              const auto &a = polygon[i];
              const auto &b = polygon[(i + 1) % n];
              const auto &c = polygon[(i + n - 1) % n];
              double uu = 0, vv = 0, uv = 0;
              for (std::size_t k = 0; k < 3; ++k) {
                const double u = double(b[k]) - double(a[k]);
                const double v = double(c[k]) - double(a[k]);
                uu += u * u;
                vv += v * v;
                uv += u * v;
              }
              if (!(uu > 0) || !(vv > 0))
                continue;
              const double cosine =
                  std::max(-1.0, std::min(1.0, uv / std::sqrt(uu * vv)));
              const double t = std::acos(cosine);
              const double ratio = std::sqrt(std::max(uu, vv) / std::min(uu, vv));
              const double bits =
                  t > 0 ? std::log2(2 * ratio) + std::log2(2 / (t * t))
                        : double(4 * limits::max_exponent);
              worst.first = std::max(worst.first, bits);
              worst.second = std::min(worst.second, t * 180 / 3.14159265358979);
            }
            return worst;
          }),
      [](const std::pair<double, double> &x, const std::pair<double, double> &y) {
        return std::pair<double, double>{std::max(x.first, y.first),
                                         std::min(x.second, y.second)};
      },
      std::pair<double, double>{0, 180}, tf::checked);
  tet_corner_need need;
  need.corner_bits = int(std::ceil(std::min(corners.first, double(ceiling))));
  need.min_angle = corners.second;
  return need;
}

} // namespace tf::tetmesh
