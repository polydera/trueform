/**
 * @file test_cdt_refiner.cpp
 * @brief Tests for tf::cdt_refiner
 *
 * Contract: bail on inputs that would force a constraint split at build
 * (crossing constraints, point on a constraint); otherwise refine to the
 * quality floor with constraints preserved verbatim modulo recorded dyadic
 * midpoint splits. Oracles: parity-filtered area conservation, constraint
 * chain presence, split record canonical form, determinism.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/constants.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/cdt_refiner.hpp>

#include <array>
#include <cmath>
#include <vector>

using refiner_index_t = int;
using Refiner = tf::cdt_refiner<refiner_index_t, double, tf::exact::int64>;

namespace {

struct planar_input {
  std::vector<std::array<double, 2>> pts;
  std::vector<std::array<refiner_index_t, 2>> cons;

  auto add(double x, double y) -> refiner_index_t {
    pts.push_back({x, y});
    return refiner_index_t(pts.size()) - 1;
  }
  auto loop(const std::vector<refiner_index_t> &l) -> void {
    for (std::size_t i = 0; i < l.size(); ++i)
      cons.push_back({l[i], l[(i + 1) % l.size()]});
  }
  auto ring(int n, double cx, double cy, double radius, double amp, int k)
      -> void {
    std::vector<refiner_index_t> l;
    for (int i = 0; i < n; ++i) {
      double t = 2.0 * tf::pi<double> * i / n;
      double r = radius * (1.0 + amp * std::sin(k * t));
      l.push_back(add(cx + r * std::cos(t), cy + r * std::sin(t)));
    }
    loop(l);
  }
};

auto run(const planar_input &in, Refiner &r,
         const tf::cdt_refine_config &config = {}) -> bool {
  tf::points_buffer<double, 2> pts;
  pts.allocate(in.pts.size());
  for (std::size_t i = 0; i < in.pts.size(); ++i)
    pts[i] = tf::point<double, 2>{in.pts[i][0], in.pts[i][1]};
  tf::buffer<refiner_index_t> flat;
  for (auto &c : in.cons) {
    flat.push_back(c[0]);
    flat.push_back(c[1]);
  }
  return r.build(pts.points(), tf::make_edges(tf::make_range(flat)), config);
}

struct interior_stats {
  double area = 0;
  double min_quality = 1;
  std::size_t n_faces = 0;
};

auto interior(const Refiner &r) -> interior_stats {
  interior_stats s;
  auto faces = r.make_faces();
  auto labels = r.region_labels();
  auto pts = r.converted_points();
  for (std::size_t i = 0; i < std::size_t(faces.size()); ++i) {
    if (labels[i] % 2 != 1)
      continue;
    ++s.n_faces;
    auto f = faces[i];
    auto a = pts[f[0]];
    auto b = pts[f[1]];
    auto c = pts[f[2]];
    double ex = b[0] - a[0], ey = b[1] - a[1];
    double fx = c[0] - a[0], fy = c[1] - a[1];
    double area2 = std::abs(ex * fy - ey * fx);
    s.area += 0.5 * area2;
    double e2 = ex * ex + ey * ey;
    double f2 = fx * fx + fy * fy;
    double g2 = (fx - ex) * (fx - ex) + (fy - ey) * (fy - ey);
    double q = (2.0 / std::sqrt(3.0)) * area2 / std::max({e2, f2, g2});
    s.min_quality = std::min(s.min_quality, q);
  }
  return s;
}

// every input constraint, subdivided by its recorded dyadic splits, must be
// present as a chain of output edges through the expected positions
auto constraints_present(const planar_input &in, const Refiner &r) -> bool {
  auto faces = r.make_faces();
  auto pts = r.converted_points();
  std::vector<std::array<refiner_index_t, 2>> edges;
  for (std::size_t i = 0; i < std::size_t(faces.size()); ++i) {
    auto f = faces[i];
    for (int k = 0; k < 3; ++k) {
      refiner_index_t u = f[k], v = f[(k + 1) % 3];
      edges.push_back({std::min(u, v), std::max(u, v)});
    }
  }
  auto find_point = [&](double x, double y) -> refiner_index_t {
    for (std::size_t i = 0; i < pts.size(); ++i)
      if (std::abs(pts[i][0] - x) < 1e-7 && std::abs(pts[i][1] - y) < 1e-7)
        return refiner_index_t(i);
    return refiner_index_t(-1);
  };
  auto has_edge = [&](refiner_index_t u, refiner_index_t v) -> bool {
    if (u > v)
      std::swap(u, v);
    for (auto &e : edges)
      if (e[0] == u && e[1] == v)
        return true;
    return false;
  };

  auto splits = r.constraint_splits();
  for (std::size_t j = 0; j < in.cons.size(); ++j) {
    auto a = in.pts[in.cons[j][0]];
    auto b = in.pts[in.cons[j][1]];
    std::vector<double> ts = {0.0};
    for (const auto &s : splits[j])
      ts.push_back(double(s.numerator) / double(1u << s.depth));
    ts.push_back(1.0);
    for (std::size_t i = 0; i + 1 < ts.size(); ++i) {
      refiner_index_t u = find_point(a[0] + ts[i] * (b[0] - a[0]),
                                     a[1] + ts[i] * (b[1] - a[1]));
      refiner_index_t v = find_point(a[0] + ts[i + 1] * (b[0] - a[0]),
                                     a[1] + ts[i + 1] * (b[1] - a[1]));
      if (u < 0 || v < 0 || !has_edge(u, v))
        return false;
    }
  }
  return true;
}

} // namespace

TEST_CASE("cdt_refiner bails on crossing constraints", "[cdt_refiner]") {
  planar_input in;
  auto a = in.add(0, 0), b = in.add(10, 0), c = in.add(10, 10),
       d = in.add(0, 10);
  in.loop({a, b, c, d});
  in.cons.push_back({a, c});
  in.cons.push_back({b, d});
  Refiner r;
  REQUIRE_FALSE(run(in, r));
  REQUIRE_FALSE(r.ok());
}

TEST_CASE("cdt_refiner bails on a self-intersecting loop", "[cdt_refiner]") {
  planar_input in;
  in.loop({in.add(0, 0), in.add(10, 10), in.add(10, 0), in.add(0, 10)});
  Refiner r;
  REQUIRE_FALSE(run(in, r));
}

TEST_CASE("cdt_refiner bails on a point inside a constraint", "[cdt_refiner]") {
  planar_input in;
  in.loop({in.add(0, 0), in.add(10, 0), in.add(10, 10), in.add(0, 10)});
  in.add(5, 0);
  in.add(5, 5);
  Refiner r;
  REQUIRE_FALSE(run(in, r));
}

TEST_CASE("cdt_refiner accepts collinear points beyond a constraint",
          "[cdt_refiner]") {
  planar_input in;
  in.loop({in.add(0, 0), in.add(10, 0), in.add(10, 10), in.add(0, 10)});
  in.add(15, 0);
  in.add(5, 5);
  Refiner r;
  REQUIRE(run(in, r));
  REQUIRE(interior(r).area == Catch::Approx(100.0).epsilon(1e-9));
}

TEST_CASE("cdt_refiner reaches the quality floor on a wiggly region",
          "[cdt_refiner]") {
  planar_input in;
  in.ring(48, 0, 0, 10, 0.25, 7);
  Refiner r;
  REQUIRE(run(in, r));
  auto s = interior(r);
  REQUIRE(s.min_quality >= 0.3);
  REQUIRE(constraints_present(in, r));
}

TEST_CASE("cdt_refiner refines a needle fan with boundary splits",
          "[cdt_refiner]") {
  planar_input in;
  std::vector<refiner_index_t> l;
  for (int i = 0; i <= 25; ++i)
    l.push_back(in.add(4.0 * i, (i % 2) * 0.5));
  l.push_back(in.add(100, 60));
  l.push_back(in.add(0, 60));
  in.loop(l);
  Refiner r;
  REQUIRE(run(in, r));
  auto s = interior(r);
  REQUIRE(s.min_quality >= 0.3);
  REQUIRE(r.n_constraint_splits() > 0);
  REQUIRE(constraints_present(in, r));
}

TEST_CASE("cdt_refiner split records are canonical dyadics", "[cdt_refiner]") {
  planar_input in;
  in.loop({in.add(0, 0), in.add(40, 0), in.add(40, 2), in.add(0, 2)});
  Refiner r;
  REQUIRE(run(in, r));
  REQUIRE(r.n_constraint_splits() > 0);
  auto splits = r.constraint_splits();
  for (std::size_t j = 0; j < in.cons.size(); ++j) {
    double prev = 0;
    for (const auto &s : splits[j]) {
      REQUIRE(s.edge == refiner_index_t(j));
      REQUIRE(s.numerator % 2 == 1);
      REQUIRE(s.depth >= 1);
      REQUIRE(s.depth <= 6);
      double t = double(s.numerator) / double(1u << s.depth);
      REQUIRE(t > prev);
      REQUIRE(t < 1.0);
      prev = t;
    }
  }
  REQUIRE(constraints_present(in, r));
}

TEST_CASE("cdt_refiner frozen boundary never splits", "[cdt_refiner]") {
  planar_input in;
  in.loop({in.add(0, 0), in.add(40, 0), in.add(40, 2), in.add(0, 2)});
  Refiner r;
  tf::cdt_refine_config config;
  config.split_encroached = false;
  REQUIRE(run(in, r, config));
  REQUIRE(r.n_constraint_splits() == 0);
  REQUIRE(interior(r).area == Catch::Approx(80.0).epsilon(1e-9));
}

TEST_CASE("cdt_refiner conserves area with holes", "[cdt_refiner]") {
  planar_input in;
  in.ring(32, 0, 0, 10, 0.0, 1);
  in.ring(20, 0, 0, 4, 0.0, 1);
  Refiner r;
  REQUIRE(run(in, r));
  auto s = interior(r);
  double want = tf::pi<double> * (100.0 - 16.0);
  // polygonal rings underestimate the disc area; compare against the loops
  double outer = 0.5 * 32 * 100.0 * std::sin(2 * tf::pi<double> / 32);
  double inner = 0.5 * 20 * 16.0 * std::sin(2 * tf::pi<double> / 20);
  (void)want;
  REQUIRE(s.area == Catch::Approx(outer - inner).epsilon(1e-9));
  REQUIRE(s.min_quality >= 0.3);
}

TEST_CASE("cdt_refiner is deterministic", "[cdt_refiner]") {
  planar_input in;
  in.ring(48, 0, 0, 10, 0.25, 7);
  Refiner r0;
  Refiner r1;
  REQUIRE(run(in, r0));
  REQUIRE(run(in, r1));
  auto f0 = r0.make_faces();
  auto f1 = r1.make_faces();
  REQUIRE(f0.size() == f1.size());
  for (std::size_t i = 0; i < std::size_t(f0.size()); ++i)
    for (int k = 0; k < 3; ++k)
      REQUIRE(f0[i][k] == f1[i][k]);
  REQUIRE(r0.n_constraint_splits() == r1.n_constraint_splits());
}

TEST_CASE("cdt_refiner consumes integer coordinates exactly", "[cdt_refiner]") {
  // pre-scaled ints with dyadic headroom: 6 bits reserved
  tf::points_buffer<std::int64_t, 2> pts;
  std::vector<std::array<std::int64_t, 2>> data = {
      {0, 0}, {40 << 6, 0}, {40 << 6, 2 << 6}, {0, 2 << 6}};
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i)
    pts[i] = tf::point<std::int64_t, 2>{data[i][0], data[i][1]};
  tf::buffer<refiner_index_t> flat;
  for (int j = 0; j < 4; ++j) {
    flat.push_back(j);
    flat.push_back((j + 1) % 4);
  }
  tf::cdt_refiner<refiner_index_t, std::int64_t, tf::exact::int64> r;
  REQUIRE(r.build(pts.points(), tf::make_edges(tf::make_range(flat))));
  REQUIRE(r.n_constraint_splits() > 0);

  // dyadic split positions land exactly on the headroom lattice
  auto out = r.converted_points();
  auto splits = r.constraint_splits();
  for (int j = 0; j < 4; ++j) {
    auto a = data[std::size_t(flat[2 * j])];
    auto b = data[std::size_t(flat[2 * j + 1])];
    for (const auto &s : splits[std::size_t(j)]) {
      std::int64_t den = std::int64_t(1) << s.depth;
      std::int64_t x = a[0] + (b[0] - a[0]) * s.numerator / den;
      std::int64_t y = a[1] + (b[1] - a[1]) * s.numerator / den;
      bool found = false;
      for (std::size_t i = 0; i < out.size() && !found; ++i)
        found = out[i][0] == x && out[i][1] == y;
      REQUIRE(found);
    }
  }
}

TEST_CASE("cdt_refiner clamps min_quality to the termination bound",
          "[cdt_refiner]") {
  planar_input in;
  in.ring(48, 0, 0, 10, 0.25, 7);
  Refiner r;
  tf::cdt_refine_config config;
  config.min_quality = 0.9f;
  REQUIRE(run(in, r, config));
  REQUIRE(interior(r).min_quality >= 0.4);
}
