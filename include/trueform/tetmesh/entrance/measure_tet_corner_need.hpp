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
#include "../../core/checked.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "./tet_corner_need.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::tetmesh {

/// The frame depth the corners of `polygons` ask and the narrowest of
/// them. A shell of radius `r` about a corner of angle `t` holds its pieces
/// equal to `t^2 / 2` of `r`, and a frame of `b` bits places a split on a
/// piece of length `l` to `2^-b l`, so a corner asks
///
///   b >= log2(l / r) + log2(2 / t^2)
///
/// with `l` the longer and `2 r` the shorter of the corner's two edges;
/// `corner_bits` is the most any corner asks, its corners read where the
/// frame `polygons` carries places them. The polygons may be the input's or
/// the entrance's, whose cuts make corners of their own. Both facts are
/// reported, never chosen by. A corner is evaluated only where the bound
/// `t^2 >= 2 (1 - cos t)` lets it pass the most its block has found, and the
/// narrowest angle is read once off the largest cosine, so the values are
/// the corner-by-corner evaluation's without its transcendentals per corner.
template <typename Policy>
auto measure_tet_corner_need(const tf::polygons<Policy> &polygons)
    -> tet_corner_need {
  using limits = std::numeric_limits<tf::coordinate_type<Policy>>;
  const double ceiling = double(4 * limits::max_exponent);
  const auto pose = tf::frame_of(polygons);
  struct local_t {
    double bits = 0;
    double bound = 1;
    double cosine = -2;
  };
  local_t need;
  tf::blocked_reduce(
      polygons, need, local_t{},
      [&pose, ceiling](const auto &range, local_t &local) {
        for (const auto &polygon : range) {
          const std::size_t n = polygon.size();
          for (std::size_t i = 0; i < n; ++i) {
            const auto a = tf::transformed(polygon[i], pose);
            const auto b = tf::transformed(polygon[(i + 1) % n], pose);
            const auto c = tf::transformed(polygon[(i + n - 1) % n], pose);
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
            local.cosine = std::max(local.cosine, cosine);
            const double ratio = std::sqrt(std::max(uu, vv) / std::min(uu, vv));
            // 4 ratio / t^2 <= 2 ratio / (1 - cos t): a corner below the
            // block's best is never evaluated
            if (cosine < 1 &&
                2 * ratio / (1 - cosine) <= local.bound * (1 - 1e-9))
              continue;
            const double t = std::acos(cosine);
            const double bits =
                t > 0 ? std::log2(2 * ratio) + std::log2(2 / (t * t)) : ceiling;
            if (bits > local.bits) {
              local.bits = bits;
              local.bound = std::exp2(bits);
            }
          }
        }
      },
      [](const local_t &local, local_t &result) {
        result.bits = std::max(result.bits, local.bits);
        result.cosine = std::max(result.cosine, local.cosine);
      },
      tf::checked);
  tet_corner_need corner;
  corner.corner_bits = int(std::ceil(std::min(need.bits, ceiling)));
  corner.min_angle =
      need.cosine < -1 ? 180 : std::acos(need.cosine) * 180 / 3.14159265358979;
  return corner;
}

} // namespace tf::tetmesh
