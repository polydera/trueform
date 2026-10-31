/**
 * @file test_hole_chart.cpp
 * @brief Tests for the chart tier's placement operations
 *
 * Tests for:
 * - tf::fill::gather_hole_collar
 * - tf::fill::make_hole_chart / tf::fill::admit_hole_chart
 * - tf::fill::fit_hole_trend / tf::fill::hole_trend_height
 * - tf::fill::sample_hole_chart
 * - tf::fill::state_hole_chart_fixed_residuals
 * - tf::fill::solve_hole_chart_residual
 * - tf::fill::place_hole_chart_samples
 * - tf::fill::fit_hole_chart_residual_basis, through tf::fill::make_hole_chart_patch
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <trueform/core/buffer.hpp>
#include <trueform/core/dot.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/triangle_quality.hpp>
#include <trueform/exact/orient2d.hpp>
#include <trueform/fill/hole_fill_status.hpp>
#include <trueform/fill/holes/admit_hole_chart.hpp>
#include <trueform/fill/holes/fit_hole_trend.hpp>
#include <trueform/fill/holes/hole_chart.hpp>
#include <trueform/exact/pt_converter.hpp>
#include <trueform/fill/holes/hole_collar.hpp>
#include <trueform/fill/holes/hole_lattice.hpp>
#include <trueform/fill/holes/hole_trend.hpp>
#include <trueform/fill/holes/make_hole_chart_patch.hpp>
#include <trueform/fill/holes/place_hole_chart_samples.hpp>
#include <trueform/fill/holes/sample_hole_chart.hpp>
#include <trueform/fill/holes/solve_hole_chart_residual.hpp>
#include <trueform/fill/holes/state_hole_chart_fixed_residuals.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <trueform/topology/cdt_refine_status.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/make_filled_mesh.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <vector>

namespace {

using chart_index = int;
using chart_lattice = std::int64_t;
using chart_mesh = tf::polygons_buffer<chart_index, double, 3, 3>;
using chart_point = tf::point<double, 3>;
using chart_uv = tf::point<double, 2>;
using chart_grid = tf::point<chart_lattice, 2>;

constexpr double chart_pi = 3.14159265358979323846;

/**
 * @brief A polar band: ring 0 is the rim, rings 1..R run outward, and every
 *        face winds so its normal is (radial out) x (angle increasing).
 */
auto chart_band_mesh(int n, int rings,
                     const std::function<chart_point(int, int)> &at)
    -> chart_mesh {
  chart_mesh mesh;
  mesh.points_buffer().allocate(std::size_t(n * (rings + 1)));
  for (int j = 0; j <= rings; ++j)
    for (int k = 0; k < n; ++k)
      mesh.points()[std::size_t(j * n + k)] = at(j, k);
  mesh.faces_buffer().allocate(std::size_t(2 * n * rings));
  std::size_t face = 0;
  for (int j = 0; j < rings; ++j)
    for (int k = 0; k < n; ++k) {
      const int inner = j * n, outer = (j + 1) * n, next = (k + 1) % n;
      mesh.faces()[face][0] = inner + k;
      mesh.faces()[face][1] = outer + k;
      mesh.faces()[face++][2] = outer + next;
      mesh.faces()[face][0] = inner + k;
      mesh.faces()[face][1] = outer + next;
      mesh.faces()[face++][2] = inner + next;
    }
  return mesh;
}

/**
 * @brief The unit sphere below polar angle `polar_deg`, its cap missing; the
 *        rim is the exactly planar circle there and the rings step by one rim
 *        edge.
 */
auto chart_sphere_band(int n, double polar_deg, int rings) -> chart_mesh {
  const double t0 = polar_deg * chart_pi / 180.0;
  const double dt = 2.0 * chart_pi * std::sin(t0) / n;
  return chart_band_mesh(n, rings, [=](int j, int k) {
    const double t = t0 + j * dt, p = 2.0 * chart_pi * k / n;
    return chart_point{std::sin(t) * std::cos(p), std::sin(t) * std::sin(p),
                       std::cos(t)};
  });
}

/**
 * @brief The graph of `g` over the annulus outside the unit circle.
 */
auto chart_graph_band(int n, int rings,
                      const std::function<double(double, double)> &g)
    -> chart_mesh {
  const double dr = 2.0 * chart_pi / n;
  return chart_band_mesh(n, rings, [=](int j, int k) {
    const double rho = 1.0 + j * dr, p = 2.0 * chart_pi * k / n;
    const double x = rho * std::cos(p), y = rho * std::sin(p);
    return chart_point{x, y, g(x, y)};
  });
}

/**
 * @brief The boundary rim holding vertex 0, in the mesh's own direction.
 */
auto chart_rim_of(const chart_mesh &mesh) -> std::vector<chart_index> {
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  for (std::size_t r = 0; r < rims.size(); ++r) {
    std::vector<chart_index> rim;
    for (auto vertex : rims.vertices[r])
      rim.push_back(chart_index(vertex));
    if (std::find(rim.begin(), rim.end(), 0) != rim.end())
      return rim;
  }
  return {};
}

/**
 * @brief Whether a lattice point lies strictly inside the rim, by its exact
 *        winding number: the chart's rim winds negatively.
 */
auto chart_inside(const tf::buffer<chart_grid> &lattice, std::size_t n,
                  const chart_grid &q) -> bool {
  int winding = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const auto &a = lattice[k];
    const auto &b = lattice[(k + 1) % n];
    const int side = tf::exact::orient2d_sign(a, b, q);
    if (side == 0 && std::min(a[0], b[0]) <= q[0] &&
        q[0] <= std::max(a[0], b[0]) && std::min(a[1], b[1]) <= q[1] &&
        q[1] <= std::max(a[1], b[1]))
      return false;
    if (a[1] <= q[1]) {
      if (b[1] > q[1] && side > 0)
        ++winding;
    } else if (b[1] <= q[1] && side < 0)
      --winding;
  }
  return winding == -1;
}

/// The lattice quantum the fill would chart this mesh with.
auto chart_unit_of(const chart_mesh &mesh) -> double {
  return tf::fill::hole_lattice_unit(
      tf::exact::make_pt_converter<chart_lattice, double>(mesh.points()));
}

/**
 * @brief One hole carried through the chart's operations.
 */
struct chart_run {
  tf::fill::hole_collar_scratch<chart_index> gather;
  tf::fill::hole_collar<chart_index> collar;
  tf::fill::hole_chart<chart_lattice> chart;
  tf::hole_chart_status verdict = tf::hole_chart_status::not_attempted;
  tf::fill::hole_trend_scratch fit;
  tf::fill::hole_trend trend{};
  tf::fill::hole_chart_sample_scratch<chart_index, chart_lattice> sample;
  tf::buffer<chart_uv> samples;
  tf::buffer<std::array<chart_index, 3>> triangles;
  tf::buffer<tf::fill::hole_chart_rim_split<chart_index, chart_lattice>> splits;
  tf::buffer<double> fixed;
  tf::fill::hole_chart_solve_scratch<chart_index> solve;
  tf::buffer<double> residual;
  tf::buffer<chart_point> placed;
  bool gathered = false, charted = false, fitted = false, sampled = false,
       solved = false, stood = false;
};

/**
 * @brief Carry one hole through gather, chart, admission, fit, sampling,
 *        residual solve and placement, stopping at the first refusal.
 */
auto chart_place(const chart_mesh &mesh, chart_run &run) -> void {
  tf::face_membership<chart_index> fm;
  fm.build(mesh.polygons());
  const auto rim = chart_rim_of(mesh);
  run.gathered = tf::fill::gather_hole_collar(mesh.polygons(), fm, rim,
                                              run.gather, run.collar);
  if (!run.gathered)
    return;
  run.charted =
      tf::fill::make_hole_chart(mesh.polygons(), run.collar,
                                chart_unit_of(mesh), run.chart) ==
      tf::hole_chart_status::admitted;
  if (!run.charted)
    return;
  run.verdict =
      tf::fill::admit_hole_chart(mesh.polygons(), run.collar, run.chart);
  if (run.verdict != tf::hole_chart_status::admitted)
    return;
  run.fitted = tf::fill::fit_hole_trend(mesh.polygons(), run.collar, run.chart,
                                        run.fit, run.trend);
  if (!run.fitted)
    return;
  run.sampled =
      tf::fill::sample_hole_chart(run.collar, run.chart, false, run.sample,
                                  run.samples, run.triangles, run.splits);
  if (!run.sampled)
    return;
  run.solved = tf::fill::state_hole_chart_fixed_residuals(
                   run.collar, run.chart, run.trend, run.fixed) &&
               tf::fill::solve_hole_chart_residual(
                   run.collar, run.chart, run.fixed, run.samples,
                   run.triangles, run.solve, run.residual);
  if (!run.solved)
    return;
  run.stood = tf::fill::place_hole_chart_samples(
      run.chart, run.trend, run.samples, run.residual, run.placed);
}

/// The worst distance of the placed points from the unit sphere.
auto chart_sphere_error(const chart_run &run) -> double {
  double worst = 0.0;
  for (const auto &p : run.placed)
    worst = std::max(
        worst,
        std::abs(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 1.0));
  return worst;
}

/**
 * @brief The plane z = 0 outside a five-lobed star rim whose vertices crowd
 *        on one side, the rings scaling the rim outward.
 */
auto chart_star_band(int n, int rings) -> chart_mesh {
  const double dr = 2.0 * chart_pi / n;
  return chart_band_mesh(n, rings, [=](int j, int k) {
    const double s = 2.0 * chart_pi * k / n;
    const double p = s + 0.8 * std::sin(3.0 * s) / 3.0;
    const double rho = (1.0 + 0.4 * std::cos(5.0 * p)) * (1.0 + j * dr);
    return chart_point{rho * std::cos(p), rho * std::sin(p), 0.0};
  });
}

/**
 * @brief The plane z = 0 outside the unit circle, rim vertex 1 pulled to a
 *        twentieth of an edge from vertex 0: a short rim edge beside a long
 *        one, which a refinement free to split the rim would split.
 */
auto chart_pinched_band(int n, int rings) -> chart_mesh {
  const double dr = 2.0 * chart_pi / n;
  return chart_band_mesh(n, rings, [=](int j, int k) {
    const double p = 2.0 * chart_pi * (j == 0 && k == 1 ? 0.05 : double(k)) / n;
    const double rho = 1.0 + j * dr;
    return chart_point{rho * std::cos(p), rho * std::sin(p), 0.0};
  });
}

/// The chart point at position `p` of `[rim | samples]`.
auto chart_position(const chart_run &run, chart_index p) -> chart_uv {
  const std::size_t n = run.collar.n_rim();
  return std::size_t(p) < n ? chart_uv{run.chart.uv[std::size_t(p)][0],
                                       run.chart.uv[std::size_t(p)][1]}
                            : run.samples[std::size_t(p) - n];
}

/// Twice the signed area of one sampled triangle, in the chart.
auto chart_area2(const chart_run &run, const std::array<chart_index, 3> &t)
    -> double {
  const auto a = chart_position(run, t[0]);
  const auto b = chart_position(run, t[1]);
  const auto c = chart_position(run, t[2]);
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

} // namespace

TEST_CASE("hole chart collar gathers the rim and two rings by membership",
          "[fill][hole_chart]") {
  const int n = 24;
  const auto mesh = chart_sphere_band(n, 50.0, 4);
  tf::face_membership<chart_index> fm;
  fm.build(mesh.polygons());
  const auto rim = chart_rim_of(mesh);
  tf::fill::hole_collar_scratch<chart_index> scratch;
  tf::fill::hole_collar<chart_index> collar;
  REQUIRE(
      tf::fill::gather_hole_collar(mesh.polygons(), fm, rim, scratch, collar));

  REQUIRE(collar.vertex_offsets ==
          std::array<chart_index, 4>{0, n, 2 * n, 3 * n});
  REQUIRE(collar.face_offsets == std::array<chart_index, 3>{0, 2 * n, 4 * n});
  for (int k = 0; k < n; ++k)
    REQUIRE(collar.vertices[std::size_t(k)] == rim[std::size_t(k)]);
  for (int k = n; k < 3 * n; ++k)
    REQUIRE(collar.vertices[std::size_t(k)] == k);
  REQUIRE(collar.ring_triangles.size() == std::size_t(2 * n));
  for (std::size_t f = 0; f < collar.ring_triangles.size(); ++f) {
    const auto corners = mesh.faces()[std::size_t(collar.faces[f])];
    for (std::size_t c = 0; c < 3; ++c)
      REQUIRE(collar.vertices[std::size_t(collar.ring_triangles[f][c])] ==
              corners[c]);
  }
}

TEST_CASE("hole chart continues a plane exactly", "[fill][hole_chart]") {
  chart_run run;
  chart_place(chart_graph_band(72, 4, [](double, double) { return 0.0; }), run);
  REQUIRE(run.stood);
  REQUIRE(run.placed.size() > 0);
  for (const auto &p : run.placed) {
    REQUIRE(std::abs(p[2]) <= 1e-12);
    REQUIRE(std::hypot(p[0], p[1]) < 1.0);
  }
}

TEST_CASE("hole chart recovers a spherical cap to second order",
          "[fill][hole_chart]") {
  chart_run coarse, fine;
  chart_place(chart_sphere_band(72, 50.0, 4), coarse);
  chart_place(chart_sphere_band(144, 50.0, 4), fine);
  REQUIRE(coarse.stood);
  REQUIRE(fine.stood);
  REQUIRE(coarse.trend.kind == tf::fill::hole_trend_kind::sphere);
  const double e = coarse.chart.edge;
  const double coarse_error = chart_sphere_error(coarse);
  const double fine_error = chart_sphere_error(fine);
  REQUIRE(coarse_error < 0.1 * e * e);
  REQUIRE(fine_error < coarse_error / 3.0);
}

TEST_CASE("hole chart selects the quadratic on a saddle",
          "[fill][hole_chart]") {
  const auto saddle = [](double x, double y) { return 0.3 * (x * x - y * y); };
  chart_run run;
  chart_place(chart_graph_band(72, 4, saddle), run);
  REQUIRE(run.stood);
  REQUIRE(run.trend.kind == tf::fill::hole_trend_kind::quadratic);
  double worst = 0.0;
  for (const auto &p : run.placed)
    worst = std::max(worst, std::abs(p[2] - saddle(p[0], p[1])));
  REQUIRE(worst < 1e-6);
}

TEST_CASE("hole chart refuses a collar steeper than admitted",
          "[fill][hole_chart]") {
  chart_run run;
  chart_place(chart_sphere_band(72, 85.0, 4), run);
  REQUIRE(run.charted);
  REQUIRE(run.verdict == tf::hole_chart_status::steep);
}

TEST_CASE("hole chart refuses a collar folding over the chart",
          "[fill][hole_chart]") {
  chart_run run;
  chart_place(chart_sphere_band(72, 105.0, 4), run);
  REQUIRE(run.charted);
  REQUIRE(run.verdict == tf::hole_chart_status::folded);
}

TEST_CASE("hole chart holds a noisy spherical cap", "[fill][hole_chart]") {
  auto mesh = chart_sphere_band(72, 50.0, 4);
  unsigned long long state = 12345;
  for (std::size_t i = 0; i < mesh.points().size(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        double((state >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
    const double s = 1.0 + 2e-3 * (2.0 * u - 1.0);
    for (std::size_t axis = 0; axis < 3; ++axis)
      mesh.points()[i][axis] *= s;
  }
  chart_run run;
  chart_place(mesh, run);
  REQUIRE(run.stood);
  REQUIRE(run.placed.size() > 0);
  REQUIRE(chart_sphere_error(run) < 1e-2);
}

TEST_CASE("hole chart residual is discretely biharmonic on the samples",
          "[fill][hole_chart]") {
  chart_run run;
  chart_place(chart_sphere_band(20, 50.0, 4), run);
  REQUIRE(run.stood);

  const std::size_t n = run.collar.n_rim();
  const std::size_t count = run.samples.size();
  const std::size_t ring = std::size_t(run.collar.vertex_offsets[2]) - n;
  const std::size_t total = n + count + ring;
  std::vector<chart_uv> at(total);
  std::vector<double> value(total, 0.0);
  const auto fixed = [&](std::size_t position, std::size_t slot) {
    double trended = 0.0;
    REQUIRE(tf::fill::hole_trend_height(run.trend, run.chart,
                                        run.chart.uv[position], trended));
    at[slot] = run.chart.uv[position];
    value[slot] = run.chart.heights[position] - trended;
  };
  for (std::size_t k = 0; k < n; ++k)
    fixed(k, k);
  for (std::size_t j = 0; j < count; ++j) {
    at[n + j] = run.samples[j];
    value[n + j] = run.residual[j];
  }
  for (std::size_t i = 0; i < ring; ++i)
    fixed(n + i, n + count + i);

  std::vector<std::array<std::size_t, 3>> triangles;
  for (const auto &t : run.triangles)
    triangles.push_back(
        {std::size_t(t[0]), std::size_t(t[1]), std::size_t(t[2])});
  for (const auto &t : run.collar.ring_triangles) {
    std::array<std::size_t, 3> lifted;
    for (std::size_t c = 0; c < 3; ++c)
      lifted[c] =
          std::size_t(t[c]) < n ? std::size_t(t[c]) : std::size_t(t[c]) + count;
    triangles.push_back(lifted);
  }
  std::vector<double> laplacian(total * total, 0.0), mass(total, 0.0);
  for (const auto &t : triangles) {
    const chart_uv p[3] = {at[t[0]], at[t[1]], at[t[2]]};
    const double area2 = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                         (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    REQUIRE(area2 > 0.0);
    for (std::size_t c = 0; c < 3; ++c) {
      mass[t[c]] += area2 / 6.0;
      const std::size_t a = t[c], b = t[(c + 1) % 3];
      const auto &o = p[(c + 2) % 3];
      const double w = 0.5 *
                       ((p[c][0] - o[0]) * (p[(c + 1) % 3][0] - o[0]) +
                        (p[c][1] - o[1]) * (p[(c + 1) % 3][1] - o[1])) /
                       area2;
      laplacian[a * total + b] -= w;
      laplacian[b * total + a] -= w;
      laplacian[a * total + a] += w;
      laplacian[b * total + b] += w;
    }
  }
  std::vector<double> stretch(total, 0.0);
  for (std::size_t i = 0; i < total; ++i)
    for (std::size_t j = 0; j < total; ++j)
      stretch[i] += laplacian[i * total + j] * value[j];
  double worst = 0.0, scale = 0.0;
  for (std::size_t j = 0; j < count; ++j) {
    const std::size_t row = n + j;
    double bend = 0.0, magnitude = 0.0;
    for (std::size_t k = 0; k < total; ++k) {
      bend += laplacian[row * total + k] * stretch[k] / mass[k];
      magnitude += std::abs(laplacian[row * total + k] * stretch[k] / mass[k]);
    }
    worst = std::max(worst, std::abs(bend));
    scale = std::max(scale, magnitude);
  }
  REQUIRE(scale > 0.0);
  REQUIRE(worst <= 1e-6 * scale);

  double lifted_miss = 0.0, residual_size = 0.0;
  for (std::size_t j = 0; j < count; ++j) {
    double trended = 0.0;
    REQUIRE(tf::fill::hole_trend_height(run.trend, run.chart, run.samples[j],
                                        trended));
    const auto expected =
        run.chart.lift(run.samples[j], trended + run.residual[j]);
    const auto offset = run.placed[j] - expected;
    lifted_miss = std::max(lifted_miss, std::sqrt(tf::dot(offset, offset)));
    residual_size = std::max(residual_size, std::abs(run.residual[j]));
  }
  REQUIRE(residual_size > 0.0);
  REQUIRE(lifted_miss <= 1e-3 * residual_size);
}

TEST_CASE("hole chart sampler grades its samples from the rim",
          "[fill][hole_chart]") {
  chart_run half, whole;
  chart_place(chart_sphere_band(600, 50.0, 4), half);
  chart_place(chart_sphere_band(1200, 50.0, 4), whole);
  REQUIRE(half.sampled);
  REQUIRE(whole.sampled);
  const double rim = double(whole.collar.n_rim());
  const double e = whole.chart.edge;
  const double area = chart_pi * whole.chart.radius * whole.chart.radius;
  REQUIRE(half.samples.size() > 0);
  REQUIRE(double(whole.samples.size()) < 3.0 * rim);
  REQUIRE(double(whole.samples.size()) < 0.1 * area / (e * e));
  REQUIRE(double(whole.samples.size()) < 2.6 * double(half.samples.size()));
  REQUIRE(whole.stood);
}

TEST_CASE("hole chart sampler states the solve's triangulation",
          "[fill][hole_chart]") {
  const auto saddle = [](double x, double y) { return 0.3 * (x * x - y * y); };
  const chart_mesh meshes[] = {
      chart_sphere_band(72, 50.0, 4), chart_star_band(96, 4),
      chart_graph_band(72, 4, saddle), chart_pinched_band(72, 4)};
  for (const auto &mesh : meshes) {
    chart_run run;
    chart_place(mesh, run);
    REQUIRE(run.sampled);
    const std::size_t n = run.collar.n_rim();
    const std::size_t count = run.samples.size();
    REQUIRE(count > 0);

    std::vector<int> uses(n + count, 0);
    std::vector<std::array<chart_index, 2>> directed;
    for (const auto &t : run.triangles) {
      REQUIRE(chart_area2(run, t) > 0.0);
      for (std::size_t c = 0; c < 3; ++c) {
        REQUIRE(t[c] >= 0);
        REQUIRE(std::size_t(t[c]) < n + count);
        ++uses[std::size_t(t[c])];
        directed.push_back({t[c], t[(c + 1) % 3]});
      }
    }
    for (std::size_t p = 0; p < n + count; ++p)
      REQUIRE(uses[p] > 0);

    std::sort(directed.begin(), directed.end());
    REQUIRE(std::adjacent_find(directed.begin(), directed.end()) ==
            directed.end());
    std::size_t rim_edges = 0;
    for (const auto &edge : directed) {
      const bool reversed =
          std::binary_search(directed.begin(), directed.end(),
                             std::array<chart_index, 2>{edge[1], edge[0]});
      const auto a = std::size_t(edge[0]), b = std::size_t(edge[1]);
      const bool on_rim = a < n && b < n && (a + 1) % n == b;
      const bool against_rim = a < n && b < n && (b + 1) % n == a;
      REQUIRE(!on_rim);
      if (against_rim) {
        REQUIRE(!reversed);
        ++rim_edges;
      } else
        REQUIRE(reversed);
    }
    REQUIRE(rim_edges == n);
    const std::size_t edges = (directed.size() + n) / 2;
    REQUIRE(n + count + run.triangles.size() == edges + 1);

    std::vector<chart_uv> sorted(run.samples.begin(), run.samples.end());
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    for (const auto &q : run.samples)
      REQUIRE(chart_inside(run.chart.lattice, n, run.chart.converter(q)));
  }
}

TEST_CASE("hole chart sampler meets its quality floor", "[fill][hole_chart]") {
  const auto saddle = [](double x, double y) { return 0.3 * (x * x - y * y); };
  const chart_mesh meshes[] = {chart_sphere_band(72, 50.0, 4),
                               chart_graph_band(72, 4, saddle),
                               chart_sphere_band(1200, 50.0, 4)};
  const double floor = tf::fill::hole_chart_sample_quality;
  const double fewest = std::asin(floor * std::sqrt(3.0) / 2.0);
  for (const auto &mesh : meshes) {
    chart_run run;
    chart_place(mesh, run);
    REQUIRE(run.sampled);
    REQUIRE(run.sample.refiner.refine_status() ==
            tf::cdt_refine_status::floor_met);
    for (const auto &t : run.triangles) {
      chart_uv p[3] = {chart_position(run, t[0]), chart_position(run, t[1]),
                       chart_position(run, t[2])};
      double longest = 0.0;
      for (std::size_t c = 0; c < 3; ++c) {
        const auto &a = p[c];
        const auto &b = p[(c + 1) % 3];
        const auto &o = p[(c + 2) % 3];
        longest = std::max(longest, (b[0] - a[0]) * (b[0] - a[0]) +
                                        (b[1] - a[1]) * (b[1] - a[1]));
        const double ax = a[0] - o[0], ay = a[1] - o[1];
        const double bx = b[0] - o[0], by = b[1] - o[1];
        REQUIRE(std::atan2(ax * by - ay * bx, ax * bx + ay * by) >=
                fewest - 1e-6);
      }
      REQUIRE(tf::triangle_quality(chart_area2(run, t), longest) >=
              floor - 1e-6);
    }
  }
}

TEST_CASE("hole chart sampler refuses a rim on one lattice point twice",
          "[fill][hole_chart]") {
  tf::fill::hole_collar<chart_index> collar;
  collar.vertex_offsets = {0, 4, 4, 4};
  tf::fill::hole_chart<chart_lattice> chart;
  chart.lattice.allocate(4);
  chart.lattice[0] = chart_grid{0, 0};
  chart.lattice[1] = chart_grid{0, 1000};
  chart.lattice[2] = chart_grid{0, 0};
  chart.lattice[3] = chart_grid{1000, 0};
  tf::fill::hole_chart_sample_scratch<chart_index, chart_lattice> scratch;
  tf::buffer<chart_uv> samples;
  tf::buffer<std::array<chart_index, 3>> triangles;
  tf::buffer<tf::fill::hole_chart_rim_split<chart_index, chart_lattice>> splits;
  REQUIRE(!tf::fill::sample_hole_chart(collar, chart, false, scratch, samples,
                                       triangles, splits));
  REQUIRE(samples.size() == 0);
  REQUIRE(triangles.size() == 0);
}

namespace {

/// The chart patch of the band's vertex-0 rim, made by the producer.
auto chart_patch_of(const chart_mesh &mesh,
                    tf::fill::hole_chart_scratch<chart_index, chart_lattice,
                                                 double> &scratch)
    -> tf::hole_chart_status {
  tf::face_membership<chart_index> fm;
  fm.build(mesh.polygons());
  const auto rim = chart_rim_of(mesh);
  tf::hole_refine_status refined{};
  return tf::fill::make_hole_chart_patch(mesh.polygons(), fm, rim,
                                         chart_unit_of(mesh), true, scratch,
                                         refined);
}

} // namespace

TEST_CASE("hole chart residual is answered by the basis where its "
          "certificate holds",
          "[fill][hole_chart]") {
  const auto mesh = chart_sphere_band(2000, 50.0, 4);
  tf::fill::hole_chart_scratch<chart_index, chart_lattice, double> scratch;
  REQUIRE(chart_patch_of(mesh, scratch) == tf::hole_chart_status::admitted);
  REQUIRE(scratch.samples.size() >= tf::fill::hole_chart_basis_samples);
  REQUIRE(scratch.solver == tf::fill::hole_chart_solver::basis);

  tf::buffer<double> meshed;
  REQUIRE(tf::fill::solve_hole_chart_residual(
      scratch.collar, scratch.chart, scratch.fixed, scratch.samples,
      scratch.triangles, scratch.solve, meshed));
  tf::buffer<chart_point> placed;
  REQUIRE(tf::fill::place_hole_chart_samples(scratch.chart, scratch.trend,
                                             scratch.samples, meshed, placed));
  double basis_error = 0.0, mesh_error = 0.0;
  for (std::size_t j = 0; j < placed.size(); ++j) {
    const auto &p = scratch.points[j];
    const auto &q = placed[j];
    basis_error = std::max(
        basis_error,
        std::abs(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 1.0));
    mesh_error = std::max(
        mesh_error,
        std::abs(std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) - 1.0));
  }
  REQUIRE(basis_error <= mesh_error);
}

TEST_CASE("hole chart residual falls to the mesh solve where the basis "
          "misses a crease",
          "[fill][hole_chart]") {
  const auto ridge = [](double x, double) { return -0.5 * std::abs(x); };
  const auto mesh = chart_graph_band(2000, 4, ridge);
  tf::fill::hole_chart_scratch<chart_index, chart_lattice, double> scratch;
  REQUIRE(chart_patch_of(mesh, scratch) == tf::hole_chart_status::admitted);
  REQUIRE(scratch.samples.size() >= tf::fill::hole_chart_basis_samples);
  REQUIRE(scratch.solver == tf::fill::hole_chart_solver::mesh);
}

TEST_CASE("hole chart splits a rim edge its quality asks to only where the "
          "rim may be split",
          "[fill][hole_chart]") {
  const auto mesh = chart_pinched_band(72, 4);
  const auto all = tf::make_boundary_rims(mesh.polygons());
  tf::boundary_rims<chart_index> rims;
  rims.vertices.offsets_buffer().push_back(0);
  rims.faces.offsets_buffer().push_back(0);
  for (std::size_t r = 0; r < all.size(); ++r) {
    bool holds = false;
    for (auto vertex : all.vertices[r])
      holds = holds || vertex == 0;
    if (!holds)
      continue;
    for (auto vertex : all.vertices[r])
      rims.vertices.data_buffer().push_back(vertex);
    for (auto face : all.faces[r])
      rims.faces.data_buffer().push_back(face);
    rims.vertices.offsets_buffer().push_back(
        chart_index(rims.vertices.data_buffer().size()));
    rims.faces.offsets_buffer().push_back(
        chart_index(rims.faces.data_buffer().size()));
    rims.closed.push_back(all.closed[r]);
  }
  REQUIRE(rims.size() == 1);

  tf::hole_fill_config config;
  const auto split = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(split.status[0] == tf::hole_fill_status::filled);
  REQUIRE(split.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(split.splits.size() > 0);
  REQUIRE(split.plan_faces.size() > 0);
  for (const auto &p : split.minted_points[0])
    REQUIRE(std::abs(p[2]) <= 1e-12);
  const auto filled = tf::make_filled_mesh(mesh.polygons(), split);
  REQUIRE(tf::is_manifold(filled.polygons()));

  config.split_rim = false;
  const auto strict = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(strict.status[0] == tf::hole_fill_status::filled);
  REQUIRE(strict.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(strict.splits.size() == 0);
  REQUIRE(strict.refined[0] == tf::hole_refine_status::stalled);
  REQUIRE(split.refined[0] == tf::hole_refine_status::floor_met);
}

TEST_CASE("hole chart trend keeps the model its subsamples agree on",
          "[fill][hole_chart]") {
  chart_run run;
  chart_place(chart_sphere_band(10000, 50.0, 6), run);
  REQUIRE(run.fitted);
  REQUIRE(run.collar.vertices.size() >= tf::fill::hole_trend_subsamples *
                                            tf::fill::hole_trend_subsample_rows);
  REQUIRE(run.trend.kind == tf::fill::hole_trend_kind::sphere);
  REQUIRE(run.stood);
  REQUIRE(chart_sphere_error(run) < 1e-5);
}
