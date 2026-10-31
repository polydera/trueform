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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/linalg/least_squares.hpp"
#include "../../core/point.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace tf::fill {

/// The sample count from which a chart's residual is first offered to the
/// basis. Below it the mesh solve is the cheaper of the two: measured on the
/// 50-degree cap, 2.6 against 2.8 ms at 1405 samples, while at 16886 the
/// basis answers in 5 against 39 ms.
inline constexpr std::size_t hole_chart_basis_samples = 2048;

/// The highest polynomial degree of the basis; a rim of `n` vertices takes
/// at most `n / 4`, so the basis never outnumbers its data.
inline constexpr std::size_t hole_chart_basis_degree = 24;

/// The worst miss at a fixed row the basis is admitted with, in rim edges.
inline constexpr double hole_chart_basis_misfit = 1e-3;

/// The scratch one basis fit walks on.
struct hole_chart_basis_scratch {
  tf::buffer<std::complex<double>> arnoldi;
  tf::buffer<std::complex<double>> hessenberg;
  tf::buffer<double> rows;
  tf::buffer<double> targets;
  tf::buffer<double> scales;
  tf::buffer<double> coefficients;
  tf::buffer<double> work;
  tf::buffer<std::size_t> permutation;
  tf::buffer<double> misses;
};

/// One point's row of the real Almansi basis: the real and imaginary parts
/// of every Arnoldi polynomial, then the same times the point's squared
/// radius, the polynomials evaluated through the Hessenberg recurrence that
/// built them. A subdiagonal entry is the norm the Arnoldi step divided by,
/// a real number, so the recurrence divides by it as one.
inline auto hole_chart_basis_row(const tf::buffer<std::complex<double>> &h,
                                 std::size_t degree,
                                 const std::complex<double> &z, double *row)
    -> void {
  std::array<std::complex<double>, tf::fill::hole_chart_basis_degree + 1> w;
  w[0] = 1.0;
  for (std::size_t k = 0; k < degree; ++k) {
    std::complex<double> next = z * w[k];
    for (std::size_t j = 0; j <= k; ++j)
      next -= h[k * (degree + 1) + j] * w[j];
    w[k + 1] = next / h[k * (degree + 1) + k + 1].real();
  }
  const double r2 = std::norm(z);
  const std::size_t half = 2 * degree + 1;
  row[0] = w[0].real();
  for (std::size_t k = 1; k <= degree; ++k) {
    row[2 * k - 1] = w[k].real();
    row[2 * k] = w[k].imag();
  }
  for (std::size_t c = 0; c < half; ++c)
    row[half + c] = r2 * row[c];
}

/// Answer the chart's residual from a basis of biharmonic functions, or
/// refuse and leave the mesh solve the authority.
///
/// Every biharmonic function of the plane is `Re f + |z|^2 Re g` for
/// analytic `f` and `g`, so polynomials of the chart point stand for both.
/// They are built by Vandermonde with Arnoldi — each new polynomial `z`
/// times the last, made orthonormal on the fitted rows — which keeps the
/// least-squares system as well conditioned at the highest degree as at the
/// lowest, and evaluated elsewhere through the recurrence that built them.
/// The fit reads an even subsample of the fixed rows, rows `i * fixed / m`;
/// the CERTIFICATE then reads every fixed row, and the basis answers only
/// where none misses by more than @ref tf::fill::hole_chart_basis_misfit rim
/// edges. Everything is serial or row-independent, so the answer is the same
/// at any worker count.
///
/// `fixed` is @ref tf::fill::state_hole_chart_fixed_residuals; `residual`
/// receives one height per sample, in `samples` order and model units.
template <typename Index, typename Int, typename Samples>
auto fit_hole_chart_residual_basis(const tf::fill::hole_collar<Index> &collar,
                                   const tf::fill::hole_chart<Int> &chart,
                                   const tf::buffer<double> &fixed,
                                   const Samples &samples,
                                   tf::fill::hole_chart_basis_scratch &scratch,
                                   tf::buffer<double> &residual) -> bool {
  const std::size_t n = collar.n_rim();
  const std::size_t rows = std::size_t(collar.vertex_offsets[2]);
  const std::size_t degree = std::min(tf::fill::hole_chart_basis_degree, n / 4);
  const std::size_t columns = 4 * degree + 2;
  const std::size_t m = std::min(rows, 8 * columns);
  if (!degree || m < 2 * columns)
    return false;

  double cx = 0.0, cy = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    cx += chart.uv[k][0];
    cy += chart.uv[k][1];
  }
  cx /= double(n);
  cy /= double(n);
  double reach = 0.0;
  for (std::size_t k = 0; k < rows; ++k)
    reach =
        std::max(reach, std::hypot(chart.uv[k][0] - cx, chart.uv[k][1] - cy));
  if (!(reach > 0.0) || !std::isfinite(reach))
    return false;
  const auto z_of = [cx, cy, reach](const auto &at) {
    return std::complex<double>((at[0] - cx) / reach, (at[1] - cy) / reach);
  };
  const auto fitted = [rows, m](std::size_t i) { return (i * rows) / m; };

  scratch.arnoldi.allocate(m * (degree + 1));
  scratch.hessenberg.allocate((degree + 1) * degree);
  std::fill(scratch.hessenberg.begin(), scratch.hessenberg.end(),
            std::complex<double>(0.0));
  for (std::size_t i = 0; i < m; ++i)
    scratch.arnoldi[i] = 1.0;
  for (std::size_t k = 0; k < degree; ++k) {
    auto *next = scratch.arnoldi.data() + (k + 1) * m;
    const auto *last = scratch.arnoldi.data() + k * m;
    for (std::size_t i = 0; i < m; ++i)
      next[i] = z_of(chart.uv[fitted(i)]) * last[i];
    for (std::size_t j = 0; j <= k; ++j) {
      const auto *basis = scratch.arnoldi.data() + j * m;
      std::complex<double> projection(0.0);
      for (std::size_t i = 0; i < m; ++i)
        projection += std::conj(basis[i]) * next[i];
      projection /= double(m);
      scratch.hessenberg[k * (degree + 1) + j] = projection;
      for (std::size_t i = 0; i < m; ++i)
        next[i] -= projection * basis[i];
    }
    double norm = 0.0;
    for (std::size_t i = 0; i < m; ++i)
      norm += std::norm(next[i]);
    norm = std::sqrt(norm / double(m));
    if (!(norm > 0.0) || !std::isfinite(norm))
      return false;
    scratch.hessenberg[k * (degree + 1) + k + 1] = norm;
    for (std::size_t i = 0; i < m; ++i)
      next[i] /= norm;
  }

  scratch.rows.allocate(m * columns);
  scratch.targets.allocate(m);
  scratch.scales.allocate(columns);
  std::fill(scratch.scales.begin(), scratch.scales.end(), 0.0);
  std::array<double, 4 * tf::fill::hole_chart_basis_degree + 2> row;
  for (std::size_t i = 0; i < m; ++i) {
    tf::fill::hole_chart_basis_row(scratch.hessenberg, degree,
                                   z_of(chart.uv[fitted(i)]), row.data());
    for (std::size_t c = 0; c < columns; ++c) {
      scratch.rows[i + c * m] = row[c];
      scratch.scales[c] += row[c] * row[c];
    }
    scratch.targets[i] = fixed[fitted(i)];
  }
  for (std::size_t c = 0; c < columns; ++c) {
    if (!(scratch.scales[c] > 0.0))
      return false;
    scratch.scales[c] = 1.0 / std::sqrt(scratch.scales[c]);
    for (std::size_t i = 0; i < m; ++i)
      scratch.rows[i + c * m] *= scratch.scales[c];
  }
  scratch.coefficients.allocate(columns);
  scratch.permutation.allocate(columns);
  scratch.work.allocate(
      tf::linalg::least_squares_workspace_size<double>(m, columns));
  tf::linalg::solve_least_squares(
      scratch.rows.data(), scratch.targets.data(), scratch.coefficients.data(),
      m, columns, scratch.work.data(), scratch.permutation.data());
  for (std::size_t c = 0; c < columns; ++c) {
    scratch.coefficients[c] *= scratch.scales[c];
    if (!std::isfinite(scratch.coefficients[c]))
      return false;
  }

  const auto value_at = [&scratch, degree,
                         columns](const std::complex<double> &z) {
    std::array<double, 4 * tf::fill::hole_chart_basis_degree + 2> at;
    tf::fill::hole_chart_basis_row(scratch.hessenberg, degree, z, at.data());
    double value = 0.0;
    for (std::size_t c = 0; c < columns; ++c)
      value += scratch.coefficients[c] * at[c];
    return value;
  };

  scratch.misses.allocate(rows);
  tf::parallel_for(
      std::size_t(0), rows,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k)
          scratch.misses[k] = std::abs(value_at(z_of(chart.uv[k])) - fixed[k]);
      },
      tf::checked);
  const double admitted =
      tf::fill::hole_chart_basis_misfit * chart.edge / chart.radius;
  for (std::size_t k = 0; k < rows; ++k)
    if (!(scratch.misses[k] <= admitted))
      return false;

  const std::size_t count = samples.size();
  residual.allocate(count);
  tf::parallel_for(
      std::size_t(0), count,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t j = first; j < last; ++j)
          residual[j] = value_at(z_of(samples[j])) * chart.radius;
      },
      tf::checked);
  for (std::size_t j = 0; j < count; ++j)
    if (!std::isfinite(residual[j]))
      return false;
  return true;
}

} // namespace tf::fill
