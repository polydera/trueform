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
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include "../../core/vector.hpp"
#include "./hole_chart.hpp"
#include <array>
#include <cmath>

namespace tf::fill {

/// Which global surface a trend is.
enum class hole_trend_kind { sphere, quadratic };

/// The one global surface a chart's heights stand on, in the chart's frame
/// scaled by its radius.
///
/// A sphere is the zero set of `c0 + (c1, c2, c3) . x + c4 |x|^2` over
/// `x = (p - origin) / radius`, so a plane is the sphere whose `c4` vanishes.
/// A quadratic is the height `c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2`
/// over the scaled chart point, in units of the radius.
struct hole_trend {
  hole_trend_kind kind;
  std::array<double, 6> coefficients;
};

/// The trend's height above chart point `at`. Returns false where the trend
/// does not cover the point, or stands there steeper than
/// @ref tf::fill::hole_chart_slope_cosine.
template <typename Int>
auto hole_trend_height(const tf::fill::hole_trend &trend,
                       const tf::fill::hole_chart<Int> &chart,
                       const tf::point<double, 2> &at, double &height) -> bool {
  const auto &c = trend.coefficients;
  const double u = at[0] / chart.radius, v = at[1] / chart.radius;
  if (trend.kind == tf::fill::hole_trend_kind::quadratic) {
    const double du = c[1] + 2.0 * c[3] * u + c[4] * v;
    const double dv = c[2] + c[4] * u + 2.0 * c[5] * v;
    height = (c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v +
              c[5] * v * v) *
             chart.radius;
    return std::isfinite(height) &&
           1.0 >= tf::fill::hole_chart_slope_cosine *
                      std::sqrt(1.0 + du * du + dv * dv);
  }
  const auto base = chart.u_axis * u + chart.v_axis * v;
  const tf::vector<double, 3> linear{c[1], c[2], c[3]};
  const double b = tf::dot(linear, chart.normal);
  const double constant =
      c[0] + tf::dot(linear, base) + c[4] * tf::dot(base, base);
  const double discriminant = b * b - 4.0 * c[4] * constant;
  if (!(discriminant >= 0.0))
    return false;
  const double denominator = b + std::sqrt(discriminant);
  if (!(denominator > 0.0))
    return false;
  const double lifted = -2.0 * constant / denominator;
  const auto gradient = linear + (base + chart.normal * lifted) * (2.0 * c[4]);
  height = lifted * chart.radius;
  return std::isfinite(height) &&
         tf::dot(gradient, chart.normal) >=
             tf::fill::hole_chart_slope_cosine *
                 std::sqrt(tf::dot(gradient, gradient));
}

} // namespace tf::fill
