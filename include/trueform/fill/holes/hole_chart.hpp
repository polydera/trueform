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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/pt_converter.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_collar.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::fill {

/// The steepest surface the chart admits, as the cosine between a surface
/// normal and the chart's own: seventy degrees. Past it a height over the
/// chart stops reading the surface, so the collar refuses and the trend does
/// not cover.
inline constexpr double hole_chart_slope_cosine = 0.3420201433256687;

/// One hole read as a height field over the plane its rim spans.
///
/// THE CHART TIER separates where a sample stands in the plane from where it
/// stands above it. The rim's Newell plane is the chart; every collar vertex
/// projects onto it and is snapped onto the MESH'S OWN LATTICE — the quantum
/// the fill's input converter states, measured from the rim's centroid —
/// and that lattice is what every decision about the chart reads exactly:
/// the rim's area, a fold of the collar, the samples and their
/// triangulation, which one quality refinement of the rim states together.
/// One quantum serves the whole fill, as one lattice serves an arrangement:
/// preflight has already read the rim on it, and a hole holds no more
/// geometry than that lattice states of it.
///
/// THE HEADROOM LAW, measured on a spherical cap scaled to span a given
/// number of lattice units: the chart matches a chart stretched over its own
/// collar above about 1e3 lattice units per rim edge, stays within 1.5 times
/// its error from about 30, and degrades below about 10, where a rim edge of
/// a few units stands at the input's own resolution and the lid or a refusal
/// is the honest answer. Above the chart the
/// height is a TREND, one global sphere or quadratic fitted to the collar,
/// plus a RESIDUAL that is biharmonic on the samples' triangulation joined to
/// the ring-1 host faces, the rim and ring 1 holding it at the host's own
/// heights. The trend carries the shape the rim alone cannot state; the
/// residual makes the rim exact and its first ring continuous.
///
/// THE COST LAW: the collar is gathered in its own size times its logarithm,
/// the trend is fitted in its size into fixed five- and six-column systems,
/// the samples grade away from the rim's own edges, so their count follows
/// the rim and not the hole's area, the residual solve costs the operator's
/// nonzeros times the iterations it runs, and the samples are evaluated once
/// each. No pass pairs every sample with every rim vertex.
///
/// `normal` is the patch's own side, the rim's Newell normal turned against
/// the host's winding, and `u_axis`, `v_axis`, `normal` are right-handed.
/// `radius` is the rim's mean distance from `origin` and `edge` its mean edge
/// length. `lattice`, `uv` and `heights` are lanes over the collar's
/// positions: the lattice point, that point read back, and the vertex's own
/// height above the chart.
///
/// @tparam Int The lattice type the chart's exact predicates dispatch on.
template <typename Int> struct hole_chart {
  tf::point<double, 3> origin;
  tf::vector<double, 3> u_axis;
  tf::vector<double, 3> v_axis;
  tf::vector<double, 3> normal;
  double radius;
  double edge;
  tf::exact::pt_converter<Int, double, 2> converter;
  tf::buffer<tf::point<Int, 2>> lattice;
  tf::buffer<tf::point<double, 2>> uv;
  tf::buffer<double> heights;

  /// The point standing `height` above chart point `at`.
  auto lift(const tf::point<double, 2> &at, double height) const
      -> tf::point<double, 3> {
    return origin + u_axis * at[0] + v_axis * at[1] + normal * height;
  }
};

/// State the chart of one gathered collar.
///
/// `unit` is the fill's lattice quantum (@ref tf::fill::hole_lattice_unit).
/// Returns `no_area` where the rim spans no plane and `out_of_lattice` where
/// a collar vertex stands farther along a chart axis than the lattice states.
template <typename Policy, typename Index, typename Int>
auto make_hole_chart(const tf::polygons<Policy> &polygons,
                     const tf::fill::hole_collar<Index> &collar, double unit,
                     tf::fill::hole_chart<Int> &chart)
    -> tf::hole_chart_status {
  const auto points = polygons.points();
  const std::size_t n = collar.n_rim();
  if (n < 3)
    return tf::hole_chart_status::no_area;
  const auto at = [&points, &collar](std::size_t position) {
    return points[std::size_t(collar.vertices[position])].template as<double>();
  };

  tf::point<double, 3> origin{0.0, 0.0, 0.0};
  for (std::size_t k = 0; k < n; ++k)
    origin = origin + at(k).as_vector();
  for (std::size_t axis = 0; axis < 3; ++axis)
    origin[axis] /= double(n);
  tf::vector<double, 3> newell{0.0, 0.0, 0.0};
  double perimeter = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const auto from = at(k) - origin;
    const auto to = at(k + 1 == n ? 0 : k + 1) - origin;
    newell = newell + tf::cross(from, to);
    const auto side = to - from;
    perimeter += std::sqrt(tf::dot(side, side));
  }
  const double area = std::sqrt(tf::dot(newell, newell));
  if (!std::isfinite(area) || area <= 0.0)
    return tf::hole_chart_status::no_area;

  chart.origin = origin;
  chart.normal = newell * (-1.0 / area);
  std::size_t flattest = 0;
  for (std::size_t axis = 1; axis < 3; ++axis)
    if (std::abs(chart.normal[axis]) < std::abs(chart.normal[flattest]))
      flattest = axis;
  tf::vector<double, 3> seed{0.0, 0.0, 0.0};
  seed[flattest] = 1.0;
  const auto across = seed - chart.normal * tf::dot(seed, chart.normal);
  chart.u_axis = across * (1.0 / std::sqrt(tf::dot(across, across)));
  chart.v_axis = tf::cross(chart.normal, chart.u_axis);

  double radius = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const auto offset = at(k) - origin;
    radius += std::sqrt(tf::dot(offset, offset));
  }
  chart.radius = radius / double(n);
  chart.edge = perimeter / double(n);

  const std::size_t count = collar.vertices.size();
  chart.uv.allocate(count);
  chart.heights.allocate(count);
  const double limit = 0.99 * double(std::numeric_limits<Int>::max());
  bool in_lattice = true;
  tf::blocked_reduce(
      tf::make_sequence_range(count), in_lattice,
      [&](const auto &block, bool &local) {
        for (std::size_t k : block) {
          const auto offset = at(k) - origin;
          chart.uv[k] = tf::point<double, 2>{tf::dot(offset, chart.u_axis),
                                             tf::dot(offset, chart.v_axis)};
          chart.heights[k] = tf::dot(offset, chart.normal);
          local = local && std::abs(chart.uv[k][0] * unit) < limit &&
                  std::abs(chart.uv[k][1] * unit) < limit;
        }
      },
      [](bool local, bool &all) { all = all && local; }, tf::checked);
  if (!in_lattice)
    return tf::hole_chart_status::out_of_lattice;
  chart.converter = tf::exact::pt_converter<Int, double, 2>{{unit, {0.0, 0.0}}};
  chart.lattice.allocate(count);
  tf::parallel_for_each(
      tf::make_sequence_range(count),
      [&chart](std::size_t k) {
        chart.lattice[k] = chart.converter(chart.uv[k]);
        chart.uv[k] = chart.converter.deconvert(chart.lattice[k]);
      },
      tf::checked);
  return tf::hole_chart_status::admitted;
}

} // namespace tf::fill
