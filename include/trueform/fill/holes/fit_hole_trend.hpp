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
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/linalg/least_squares.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/vector.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include "./hole_trend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::fill {

/// The scratch one trend fit walks on: the weighted rows, column-major, and
/// the least-squares kernel's own workspace.
struct hole_trend_scratch {
  tf::buffer<double> rows;
  tf::buffer<double> targets;
  tf::buffer<double> work;
};

/// The rows one subsample of a large collar fits, of its vertices and of its
/// faces each.
inline constexpr std::size_t hole_trend_subsample_rows = 512;

/// The disjoint subsamples a large collar is fitted on.
inline constexpr std::size_t hole_trend_subsamples = 4;

/// How far the subsample fits of one model may spread, relative to its
/// largest coefficient, and still state one surface: measured on the
/// 50-degree cap ladder, the fits of the exact model agree to 6e-5 at a
/// 10000-vertex rim and closer elsewhere, while a model one subsample can
/// corrupt spreads past it.
inline constexpr double hole_trend_agreement = 1e-4;

/// Fit both trends to the collar and keep the one whose worst miss at the rim
/// is smaller, the sphere on a tie.
///
/// A collar of at least @ref tf::fill::hole_trend_subsamples times
/// @ref tf::fill::hole_trend_subsample_rows rows of each kind is fitted on as
/// many disjoint even subsamples — rows `j`, `j + step`, ... — and a model
/// whose fits agree within @ref tf::fill::hole_trend_agreement states their
/// mean. A model whose fits disagree is one a subsample can corrupt and does
/// not compete; where neither agrees, the whole collar is fitted. A smaller
/// collar is always fitted whole.
///
/// Every collar vertex is a position row of weight one, and every collar face
/// states its unit normal at its centroid in rows of weight `(edge /
/// radius)^2`, so a normal off by an angle weighs as a position off by that
/// angle times an edge. The quadratic reads a face's normal as a slope and so
/// takes only the faces the chart admits. Returns false where neither trend
/// covers the whole rim.
template <typename Policy, typename Index, typename Int>
auto fit_hole_trend(const tf::polygons<Policy> &polygons,
                    const tf::fill::hole_collar<Index> &collar,
                    const tf::fill::hole_chart<Int> &chart,
                    tf::fill::hole_trend_scratch &scratch,
                    tf::fill::hole_trend &trend) -> bool {
  const auto points = polygons.points();
  const auto faces = polygons.faces();
  const std::size_t n_vertices = collar.vertices.size();
  const std::size_t n_faces = collar.faces.size();
  const double scale = 1.0 / chart.radius;
  const double slope_weight = chart.edge * scale;

  const auto scaled = [&chart, scale](const tf::point<double, 3> &p) {
    return (p - chart.origin) * scale;
  };
  const auto face_at = [&](std::size_t f, tf::point<double, 3> &centroid,
                           tf::vector<double, 3> &normal) {
    const auto corners = faces[std::size_t(collar.faces[f])];
    const auto p0 = points[std::size_t(corners[0])].template as<double>();
    const auto p1 = points[std::size_t(corners[1])].template as<double>();
    const auto p2 = points[std::size_t(corners[2])].template as<double>();
    centroid = p0 + ((p1 - p0) + (p2 - p0)) * (1.0 / 3.0);
    normal = tf::cross(p1 - p0, p2 - p0);
    const double length = std::sqrt(tf::dot(normal, normal));
    if (!(length > 0.0))
      return false;
    normal = normal * (1.0 / length);
    return true;
  };

  const auto solve = [&scratch](std::size_t count, std::size_t columns,
                                std::array<double, 6> &coefficients) {
    scratch.work.allocate(
        tf::linalg::least_squares_workspace_size<double>(count, columns));
    std::array<std::size_t, 6> permutation;
    coefficients = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    tf::linalg::solve_least_squares(scratch.rows.data(), scratch.targets.data(),
                                    coefficients.data(), count, columns,
                                    scratch.work.data(), permutation.data());
  };

  const auto rim_miss = [&chart, &collar](const tf::fill::hole_trend &fit) {
    double worst = 0.0;
    for (std::size_t k = 0; k < collar.n_rim(); ++k) {
      double height = 0.0;
      if (!tf::fill::hole_trend_height(fit, chart, chart.uv[k], height))
        return std::numeric_limits<double>::infinity();
      worst = std::max(worst, std::abs(chart.heights[k] - height));
    }
    return worst;
  };

  const auto sampled = [](std::size_t n, std::size_t first, std::size_t step) {
    return first >= n ? std::size_t(0) : (n - first + step - 1) / step;
  };
  const auto fit_rows = [&](std::size_t first, std::size_t step_vertices,
                            std::size_t step_faces,
                            tf::fill::hole_trend &sphere,
                            tf::fill::hole_trend &quadratic) {
    sphere = {tf::fill::hole_trend_kind::sphere, {}};
    {
      const std::size_t count = sampled(n_vertices, first, step_vertices) +
                                3 * sampled(n_faces, first, step_faces);
      scratch.rows.allocate(count * 5);
      scratch.targets.allocate(count);
      std::size_t row = 0;
      const auto put = [&scratch, count, &row](std::array<double, 5> entries,
                                               double target, double weight) {
        for (std::size_t column = 0; column < 5; ++column)
          scratch.rows[row + column * count] = weight * entries[column];
        scratch.targets[row++] = weight * target;
      };
      for (std::size_t k = first; k < n_vertices; k += step_vertices) {
        const auto x = scaled(
            points[std::size_t(collar.vertices[k])].template as<double>());
        put({1.0, x[0], x[1], x[2], tf::dot(x, x)}, 0.0, 1.0);
      }
      for (std::size_t f = first; f < n_faces; f += step_faces) {
        tf::point<double, 3> centroid;
        tf::vector<double, 3> normal;
        const bool stated = face_at(f, centroid, normal);
        const auto x = scaled(centroid);
        for (std::size_t axis = 0; axis < 3; ++axis) {
          std::array<double, 5> entries{0.0, 0.0, 0.0, 0.0, 2.0 * x[axis]};
          entries[1 + axis] = 1.0;
          put(entries, normal[axis], stated ? slope_weight : 0.0);
        }
      }
      solve(count, 5, sphere.coefficients);
    }

    quadratic = {tf::fill::hole_trend_kind::quadratic, {}};
    {
      const std::size_t count = sampled(n_vertices, first, step_vertices) +
                                2 * sampled(n_faces, first, step_faces);
      scratch.rows.allocate(count * 6);
      scratch.targets.allocate(count);
      std::size_t row = 0;
      const auto put = [&scratch, count, &row](std::array<double, 6> entries,
                                               double target, double weight) {
        for (std::size_t column = 0; column < 6; ++column)
          scratch.rows[row + column * count] = weight * entries[column];
        scratch.targets[row++] = weight * target;
      };
      for (std::size_t k = first; k < n_vertices; k += step_vertices) {
        const double u = chart.uv[k][0] * scale, v = chart.uv[k][1] * scale;
        put({1.0, u, v, u * u, u * v, v * v}, chart.heights[k] * scale, 1.0);
      }
      for (std::size_t f = first; f < n_faces; f += step_faces) {
        tf::point<double, 3> centroid;
        tf::vector<double, 3> normal;
        const bool stated = face_at(f, centroid, normal);
        const auto x = scaled(centroid);
        const double u = tf::dot(x, chart.u_axis), v = tf::dot(x, chart.v_axis);
        const double lean = tf::dot(normal, chart.normal);
        const bool admitted =
            stated && lean >= tf::fill::hole_chart_slope_cosine;
        const double weight = admitted ? slope_weight : 0.0;
        const double rise = admitted ? 1.0 / lean : 0.0;
        put({0.0, 1.0, 0.0, 2.0 * u, v, 0.0},
            -tf::dot(normal, chart.u_axis) * rise, weight);
        put({0.0, 0.0, 1.0, 0.0, u, 2.0 * v},
            -tf::dot(normal, chart.v_axis) * rise, weight);
      }
      solve(count, 6, quadratic.coefficients);
    }
  };

  const std::size_t k = tf::fill::hole_trend_subsamples;
  const std::size_t step_vertices =
      n_vertices / tf::fill::hole_trend_subsample_rows;
  const std::size_t step_faces = n_faces / tf::fill::hole_trend_subsample_rows;
  tf::fill::hole_trend sphere{}, quadratic{};
  bool stable_sphere = true, stable_quadratic = true;
  if (step_vertices < k || step_faces < k)
    fit_rows(0, 1, 1, sphere, quadratic);
  else {
    std::array<tf::fill::hole_trend, tf::fill::hole_trend_subsamples> spheres,
        quadratics;
    for (std::size_t j = 0; j < k; ++j)
      fit_rows(j, step_vertices, step_faces, spheres[j], quadratics[j]);
    const auto agree = [k](const auto &fits, tf::fill::hole_trend &mean) {
      mean = {fits[0].kind, {}};
      for (std::size_t c = 0; c < 6; ++c)
        for (std::size_t j = 0; j < k; ++j)
          mean.coefficients[c] += fits[j].coefficients[c] / double(k);
      double spread = 0.0, magnitude = 0.0;
      for (std::size_t c = 0; c < 6; ++c) {
        magnitude = std::max(magnitude, std::abs(mean.coefficients[c]));
        for (std::size_t j = 0; j < k; ++j)
          spread = std::max(
              spread, std::abs(fits[j].coefficients[c] - mean.coefficients[c]));
      }
      return magnitude > 0.0 &&
             spread <= tf::fill::hole_trend_agreement * magnitude;
    };
    stable_sphere = agree(spheres, sphere);
    stable_quadratic = agree(quadratics, quadratic);
    if (!stable_sphere && !stable_quadratic) {
      fit_rows(0, 1, 1, sphere, quadratic);
      stable_sphere = stable_quadratic = true;
    }
  }

  const double sphere_miss = stable_sphere
                                 ? rim_miss(sphere)
                                 : std::numeric_limits<double>::infinity();
  const double quadratic_miss = stable_quadratic
                                    ? rim_miss(quadratic)
                                    : std::numeric_limits<double>::infinity();
  if (!std::isfinite(std::min(sphere_miss, quadratic_miss)))
    return false;
  trend = sphere_miss <= quadratic_miss ? sphere : quadratic;
  return true;
}

} // namespace tf::fill
