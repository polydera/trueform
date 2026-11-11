/**
 * @file test_geodesic_distances.cpp
 * @brief Tests for the unfolding geodesic distances
 *
 * Tests for:
 * - tf::make_geodesic_distances
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "worker_ladder.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <trueform/core/buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/geodesic/geodesic_distances_within.hpp>
#include <trueform/geometry/geodesic/geodesic_front_stats.hpp>
#include <trueform/geometry/geodesic/propagate_geodesic_distances.hpp>
#include <trueform/geometry/make_geodesic_distances.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/vertex_link.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Shortest paths along the mesh's edges, independently of the
 *        library: what a geodesic must beat.
 */
template <typename Mesh>
auto geodesic_edge_paths(const Mesh &mesh, const std::vector<int> &seeds)
    -> std::vector<double> {
  const std::size_t n = mesh.points().size();
  std::vector<std::vector<std::pair<int, double>>> edges(n);
  for (std::size_t f = 0; f < mesh.faces().size(); ++f)
    for (int c = 0; c < 3; ++c) {
      const int a = int(mesh.faces()[f][c]);
      const int b = int(mesh.faces()[f][(c + 1) % 3]);
      double length2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double d = double(mesh.points()[std::size_t(a)][axis]) -
                         double(mesh.points()[std::size_t(b)][axis]);
        length2 += d * d;
      }
      edges[std::size_t(a)].push_back({b, std::sqrt(length2)});
      edges[std::size_t(b)].push_back({a, std::sqrt(length2)});
    }
  std::vector<double> d(n, std::numeric_limits<double>::infinity());
  std::priority_queue<std::pair<double, int>,
                      std::vector<std::pair<double, int>>, std::greater<>>
      queue;
  for (auto seed : seeds) {
    d[std::size_t(seed)] = 0.0;
    queue.push({0.0, seed});
  }
  while (!queue.empty()) {
    const auto [distance, v] = queue.top();
    queue.pop();
    if (distance > d[std::size_t(v)])
      continue;
    for (const auto &[peer, length] : edges[std::size_t(v)])
      if (distance + length < d[std::size_t(peer)]) {
        d[std::size_t(peer)] = distance + length;
        queue.push({d[std::size_t(peer)], peer});
      }
  }
  return d;
}

template <typename Point>
auto geodesic_euclid(const Point &a, const Point &b) -> double {
  double length2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double d = double(a[axis]) - double(b[axis]);
    length2 += d * d;
  }
  return std::sqrt(length2);
}

/** @brief Two copies of a plane side by side that share no point. */
auto geodesic_two_sheets() -> tf::polygons_buffer<int, double, 3, 3> {
  const auto plane = tf::make_plane_mesh<int>(2.0, 2.0, 4, 4);
  const std::size_t n = plane.points().size();
  tf::polygons_buffer<int, double, 3, 3> out;
  out.points_buffer().allocate(2 * n);
  for (std::size_t copy = 0; copy < 2; ++copy)
    for (std::size_t k = 0; k < n; ++k) {
      out.points()[copy * n + k] = plane.points()[k];
      out.points()[copy * n + k][0] += 5.0 * double(copy);
    }
  const std::size_t f = plane.faces().size();
  out.faces_buffer().allocate(2 * f);
  for (std::size_t copy = 0; copy < 2; ++copy)
    for (std::size_t t = 0; t < f; ++t)
      for (std::size_t c = 0; c < 3; ++c)
        out.faces()[copy * f + t][c] = plane.faces()[t][c] + int(copy * n);
  return out;
}

/**
 * @brief A unit plane of `ticks` squares a side whose interior points are
 *        moved by up to a quarter of the spacing, so no edge follows a ray
 *        from the centre; the centre stays put.
 */
auto geodesic_jittered_plane(int ticks)
    -> tf::polygons_buffer<int, double, 3, 3> {
  auto plane = tf::make_plane_mesh<int>(1.0, 1.0, ticks, ticks);
  const double spacing = 1.0 / double(ticks);
  auto unit = [](std::uint64_t key) {
    key += 0x9e3779b97f4a7c15ull;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebull;
    key ^= key >> 31;
    return double(key >> 11) / double(1ull << 53) * 2.0 - 1.0;
  };
  for (std::size_t k = 0; k < plane.points().size(); ++k) {
    const double x = plane.points()[k][0], y = plane.points()[k][1];
    if (std::abs(x) > 0.5 - 0.5 * spacing ||
        std::abs(y) > 0.5 - 0.5 * spacing ||
        (std::abs(x) < 0.5 * spacing && std::abs(y) < 0.5 * spacing))
      continue;
    for (int axis = 0; axis < 2; ++axis)
      plane.points()[k][axis] +=
          0.25 * spacing * unit(2 * k + std::size_t(axis));
  }
  return plane;
}

/**
 * @brief A unit square below and one above, joined by `layers` rows of needle
 *        triangles `height` tall, every other row shifted by half a cell, so
 *        the needles stand at an aspect of about `0.1 / height`.
 */
auto geodesic_needle_strip(int n, double height, int layers)
    -> tf::polygons_buffer<int, double, 3, 3> {
  std::vector<std::vector<int>> rows;
  std::vector<double> xyz;
  auto add_row = [&](double y, double shift) {
    std::vector<int> row;
    for (int i = 0; i <= n; ++i) {
      const double x = (double(i) + (i > 0 && i < n ? shift : 0.0)) / n;
      row.push_back(int(xyz.size() / 3));
      xyz.insert(xyz.end(), {x, y, 0.0});
    }
    rows.push_back(row);
  };
  for (int r = 0; r <= n; ++r)
    add_row(-1.0 + double(r) / n, 0.0);
  for (int l = 1; l <= layers; ++l)
    add_row(height * l, l % 2 ? 0.5 : 0.0);
  for (int r = 1; r <= n; ++r)
    add_row(height * layers + double(r) / n, 0.0);
  tf::polygons_buffer<int, double, 3, 3> out;
  out.points_buffer().allocate(xyz.size() / 3);
  for (std::size_t k = 0; k < xyz.size() / 3; ++k)
    for (std::size_t axis = 0; axis < 3; ++axis)
      out.points()[k][axis] = xyz[3 * k + axis];
  out.faces_buffer().allocate(2 * (rows.size() - 1) * std::size_t(n));
  std::size_t t = 0;
  for (std::size_t r = 0; r + 1 < rows.size(); ++r)
    for (std::size_t i = 0; i < std::size_t(n); ++i) {
      const int a = rows[r][i], b = rows[r][i + 1], c = rows[r + 1][i],
                d = rows[r + 1][i + 1];
      out.faces()[t][0] = a;
      out.faces()[t][1] = b;
      out.faces()[t][2] = d;
      ++t;
      out.faces()[t][0] = a;
      out.faces()[t][1] = d;
      out.faces()[t][2] = c;
      ++t;
    }
  return out;
}

} // namespace

TEST_CASE("geodesic_distances: a jittered plane answers the straight line at "
          "every refinement",
          "[geometry][geodesic]") {
  const double tolerance = tf::geometry::geodesic_revision_tolerance;
  for (int ticks : {20, 40, 80}) {
    const auto plane = geodesic_jittered_plane(ticks);
    int centre = 0;
    for (std::size_t k = 0; k < plane.points().size(); ++k)
      if (std::abs(plane.points()[k][0]) + std::abs(plane.points()[k][1]) <
          std::abs(plane.points()[std::size_t(centre)][0]) +
              std::abs(plane.points()[std::size_t(centre)][1]))
        centre = int(k);
    double edge = 0.0;
    for (std::size_t f = 0; f < plane.faces().size(); ++f)
      for (std::size_t c = 0; c < 3; ++c)
        edge += geodesic_euclid(
            plane.points()[std::size_t(plane.faces()[f][c])],
            plane.points()[std::size_t(plane.faces()[f][(c + 1) % 3])]);
    edge /= double(3 * plane.faces().size());
    // The refinement is the update's; a split front's schedule adds its own
    // tolerance, which the split case below bounds.
    tbb::task_arena one(1);
    const auto distances = one.execute([&] {
      return tf::make_geodesic_distances(plane.polygons(),
                                         std::vector<int>{centre});
    });
    // On a plane the update is exact, so what is left is the tolerance a
    // revision declines, at most its share per edge the front crossed.
    double worst = 0.0;
    for (std::size_t k = 0; k < plane.points().size(); ++k) {
      const double line = geodesic_euclid(plane.points()[k],
                                          plane.points()[std::size_t(centre)]);
      CAPTURE(ticks, k, distances[k], line, edge);
      REQUIRE(std::abs(distances[k] - line) <=
              tolerance * line * (1.0 + line / edge));
      worst = std::max(worst, std::abs(distances[k] - line));
    }
    CAPTURE(ticks, worst);
    REQUIRE(worst <= 2e-6);
  }
}

TEST_CASE("geodesic_distances: a plane answers the straight line, which the "
          "edges cannot",
          "[geometry][geodesic]") {
  const auto strip = tf::make_plane_mesh<int>(12.0, 3.0, 48, 12);
  int seed = 0;
  for (std::size_t k = 0; k < strip.points().size(); ++k)
    if (strip.points()[k][0] < strip.points()[std::size_t(seed)][0] ||
        (strip.points()[k][0] == strip.points()[std::size_t(seed)][0] &&
         strip.points()[k][1] < strip.points()[std::size_t(seed)][1]))
      seed = int(k);
  const auto distances =
      tf::make_geodesic_distances(strip.polygons(), std::vector<int>{seed});
  const auto paths = geodesic_edge_paths(strip, {seed});
  REQUIRE(distances.size() == strip.points().size());
  REQUIRE(distances[std::size_t(seed)] == 0.0);
  double worst_edge = 0.0;
  for (std::size_t k = 0; k < strip.points().size(); ++k) {
    const double line =
        geodesic_euclid(strip.points()[k], strip.points()[std::size_t(seed)]);
    REQUIRE(std::abs(distances[k] - line) <= 1e-9 * (1.0 + line));
    if (line > 1.0)
      worst_edge = std::max(worst_edge, paths[k] / line);
  }
  REQUIRE(worst_edge > 1.05);
}

TEST_CASE("geodesic_distances: a sphere answers the arc, which the edges "
          "cannot",
          "[geometry][geodesic]") {
  const double radius = 2.0;
  const auto sphere = tf::make_sphere_mesh<int>(radius, 60, 80);
  int seed = 0;
  for (std::size_t k = 0; k < sphere.points().size(); ++k)
    if (std::abs(sphere.points()[k][2]) <
        std::abs(sphere.points()[std::size_t(seed)][2]))
      seed = int(k);
  const auto distances =
      tf::make_geodesic_distances(sphere.polygons(), std::vector<int>{seed});
  const auto paths = geodesic_edge_paths(sphere, {seed});
  const auto source = sphere.points()[std::size_t(seed)];
  double worst = 0.0, worst_edge = 0.0;
  for (std::size_t k = 0; k < sphere.points().size(); ++k) {
    double inner = 0.0;
    for (int axis = 0; axis < 3; ++axis)
      inner += double(sphere.points()[k][axis]) * double(source[axis]);
    const double arc =
        radius * std::acos(std::clamp(inner / (radius * radius), -1.0, 1.0));
    if (arc < 0.5 || arc > radius * 2.5)
      continue;
    worst = std::max(worst, std::abs(distances[k] - arc) / arc);
    worst_edge = std::max(worst_edge, std::abs(paths[k] - arc) / arc);
  }
  REQUIRE(worst < 0.01);
  REQUIRE(worst_edge > 3.0 * worst);
}

TEST_CASE("geodesic_distances: several seeds answer the nearest of them",
          "[geometry][geodesic]") {
  const auto plane = tf::make_plane_mesh<int>(4.0, 4.0, 16, 16);
  std::vector<int> seeds;
  for (std::size_t k = 0; k < plane.points().size(); ++k) {
    const auto p = plane.points()[k];
    if ((p[0] == -2.0 && p[1] == -2.0) || (p[0] == 2.0 && p[1] == 2.0))
      seeds.push_back(int(k));
  }
  REQUIRE(seeds.size() == 2);
  const auto distances = tf::make_geodesic_distances(plane.polygons(), seeds);
  std::size_t apart = 0;
  for (std::size_t k = 0; k < plane.points().size(); ++k) {
    const double first = geodesic_euclid(plane.points()[k],
                                         plane.points()[std::size_t(seeds[0])]);
    const double second = geodesic_euclid(
        plane.points()[k], plane.points()[std::size_t(seeds[1])]);
    const double nearest = std::min(first, second);
    // Where the two fronts meet, a face sees one corner from each seed and
    // unfolds a source neither of them is: the answer stays near, not exact.
    if (nearest < 0.85 * std::max(first, second)) {
      REQUIRE(std::abs(distances[k] - nearest) <= 1e-9 * (1.0 + nearest));
      ++apart;
    } else
      REQUIRE(std::abs(distances[k] - nearest) <= 0.03 * nearest);
  }
  REQUIRE(apart > 200);
}

TEST_CASE("geodesic_distances: a reach answers within it and nothing past it",
          "[geometry][geodesic]") {
  const auto plane = tf::make_plane_mesh<int>(4.0, 4.0, 16, 16);
  int centre = 0;
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    if (plane.points()[k][0] == 0.0 && plane.points()[k][1] == 0.0)
      centre = int(k);
  const auto distances = tf::make_geodesic_distances(
      plane.polygons(), std::vector<int>{centre}, 1.3);
  std::size_t within = 0, past = 0;
  for (std::size_t k = 0; k < plane.points().size(); ++k) {
    const double line =
        geodesic_euclid(plane.points()[k], plane.points()[std::size_t(centre)]);
    if (line <= 1.25) {
      REQUIRE(std::abs(distances[k] - line) <= 1e-9 * (1.0 + line));
      ++within;
    } else if (line > 1.3) {
      REQUIRE(std::isinf(distances[k]));
      ++past;
    }
  }
  REQUIRE(within > 50);
  REQUIRE(past > 100);
}

TEST_CASE("geodesic_distances: a piece no seed stands on is unreached",
          "[geometry][geodesic]") {
  const auto sheets = geodesic_two_sheets();
  const auto distances =
      tf::make_geodesic_distances(sheets.polygons(), std::vector<int>{0});
  const std::size_t n = sheets.points().size() / 2;
  for (std::size_t k = 0; k < n; ++k)
    REQUIRE(std::isfinite(distances[k]));
  for (std::size_t k = n; k < 2 * n; ++k)
    REQUIRE(std::isinf(distances[k]));
}

TEST_CASE("geodesic_distances: no seed reaches nothing",
          "[geometry][geodesic]") {
  const auto plane = tf::make_plane_mesh<int>(4.0, 4.0, 4, 4);
  const auto distances =
      tf::make_geodesic_distances(plane.polygons(), std::vector<int>{});
  REQUIRE(distances.size() == plane.points().size());
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    REQUIRE(std::isinf(distances[k]));
}

TEST_CASE("geodesic_distances: tagged structures answer what built ones do",
          "[geometry][geodesic]") {
  const auto sphere = tf::make_sphere_mesh<int>(1.0, 20, 24);
  tf::face_membership<int> membership;
  membership.build(sphere.polygons());
  tf::vertex_link<int> link;
  link.build(sphere.polygons().faces(), membership);
  const auto built =
      tf::make_geodesic_distances(sphere.polygons(), std::vector<int>{7});
  const auto tagged = tf::make_geodesic_distances(
      sphere.polygons() | tf::tag(membership) | tf::tag(link),
      std::vector<int>{7});
  for (std::size_t k = 0; k < sphere.points().size(); ++k) {
    REQUIRE(std::isfinite(built[k]));
    REQUIRE(built[k] == tagged[k]);
  }
}

TEST_CASE("geodesic_distances: a split front answers within epsilon per edge "
          "crossed of one heap",
          "[geometry][geodesic][parallel]") {
  const auto plane = geodesic_jittered_plane(160);
  const auto sphere = tf::make_sphere_mesh<int>(1.0, 120, 160);
  int centre = 0;
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    if (std::abs(plane.points()[k][0]) + std::abs(plane.points()[k][1]) <
        std::abs(plane.points()[std::size_t(centre)][0]) +
            std::abs(plane.points()[std::size_t(centre)][1]))
      centre = int(k);
  const double epsilon = tf::geometry::geodesic_revision_tolerance;
  auto agree = [epsilon](const auto &mesh, const std::vector<int> &seeds,
                         double max_distance) {
    double edge = 0.0, longest = 0.0;
    for (std::size_t f = 0; f < mesh.faces().size(); ++f)
      for (std::size_t c = 0; c < 3; ++c) {
        const double length = geodesic_euclid(
            mesh.points()[std::size_t(mesh.faces()[f][c])],
            mesh.points()[std::size_t(mesh.faces()[f][(c + 1) % 3])]);
        edge += length;
        longest = std::max(longest, length);
      }
    edge /= double(3 * mesh.faces().size());
    tbb::task_arena one(1);
    tf::geometry::geodesic_front_stats heap_stats;
    const auto heap = one.execute([&] {
      return tf::geometry::geodesic_distances_within<double>(
          mesh.polygons(), seeds, max_distance, heap_stats);
    });
    const std::size_t workers = tf::test::capped_workers(8);
    const tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, workers);
    tbb::task_arena several(static_cast<int>(workers));
    tf::geometry::geodesic_front_stats split_stats;
    const auto split = several.execute([&] {
      return tf::geometry::geodesic_distances_within<double>(
          mesh.polygons(), seeds, max_distance, split_stats);
    });
    REQUIRE(split.size() == heap.size());
    // One heap is the reference, and the bound below says nothing unless
    // the several-worker run is the split front it is stated against.
    REQUIRE(heap_stats.fronts == 1);
    REQUIRE((split_stats.fronts > 1) == (workers > 1));
    std::size_t compared = 0;
    for (std::size_t k = 0; k < heap.size(); ++k) {
      // Near the reach a point reads corners past it, and the clamp to
      // infinity turns an epsilon into a jump.
      if (!(heap[k] < max_distance - 4.0 * edge))
        continue;
      // Where two seeds' fronts meet, which seed states a point is the
      // schedule's, and the seam answers the edge from either side. A point
      // reads a seam point's seed through one face, whose edges move each
      // seed's distance by at most the longest, so the schedule reaches no
      // point whose two nearest seeds differ by twice the longest edge. The
      // several-seed case runs on a plane, where the line is that distance.
      double first = std::numeric_limits<double>::infinity(), second = first;
      for (const int seed : seeds) {
        const double line =
            geodesic_euclid(mesh.points()[k], mesh.points()[std::size_t(seed)]);
        second = std::min(second, std::max(first, line));
        first = std::min(first, line);
      }
      if (second - first < 2.0 * longest)
        continue;
      REQUIRE(std::isfinite(split[k]));
      CAPTURE(k, heap[k], split[k], edge);
      REQUIRE(std::abs(split[k] - heap[k]) <=
              epsilon * heap[k] * (1.0 + heap[k] / edge));
      ++compared;
    }
    REQUIRE(compared > heap.size() / 4);
  };
  agree(plane, {centre}, std::numeric_limits<double>::infinity());
  agree(plane, {centre, 0}, std::numeric_limits<double>::infinity());
  agree(plane, {centre}, 0.4);
  agree(sphere, {7}, std::numeric_limits<double>::infinity());
}

TEST_CASE("geodesic_distances: under a global limit of one the front runs "
          "on one heap and answers its bits",
          "[geometry][geodesic]") {
  const auto plane = geodesic_jittered_plane(40);
  int centre = 0;
  for (std::size_t k = 0; k < plane.points().size(); ++k)
    if (std::abs(plane.points()[k][0]) + std::abs(plane.points()[k][1]) <
        std::abs(plane.points()[std::size_t(centre)][0]) +
            std::abs(plane.points()[std::size_t(centre)][1]))
      centre = int(k);
  const std::vector<int> seeds{centre};
  const double unbounded = std::numeric_limits<double>::infinity();
  tbb::task_arena one(1);
  tf::geometry::geodesic_front_stats heap_stats;
  const auto heap = one.execute([&] {
    return tf::geometry::geodesic_distances_within<double>(
        plane.polygons(), seeds, unbounded, heap_stats);
  });
  REQUIRE(heap_stats.fronts == 1);
  // The plane is past the split threshold: at the machine's width the front
  // splits wherever there is more than one worker.
  tf::geometry::geodesic_front_stats wide_stats;
  tf::geometry::geodesic_distances_within<double>(plane.polygons(), seeds,
                                                  unbounded, wide_stats);
  REQUIRE((wide_stats.fronts > 1) ==
          (tbb::this_task_arena::max_concurrency() > 1));

  const tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                                  1);
  tf::geometry::geodesic_front_stats limited_stats;
  const auto limited = tf::geometry::geodesic_distances_within<double>(
      plane.polygons(), seeds, unbounded, limited_stats);
  REQUIRE(limited_stats.fronts == 1);
  REQUIRE(limited.size() == heap.size());
  REQUIRE(std::memcmp(limited.begin(), heap.begin(),
                      heap.size() * sizeof(double)) == 0);
}

TEST_CASE("geodesic_distances: the distances are stated in the resolved "
          "output type",
          "[geometry][geodesic]") {
  const auto sphere = tf::make_sphere_mesh<int>(1.0f, 20, 24);
  const auto own =
      tf::make_geodesic_distances(sphere.polygons(), std::vector<int>{7});
  const auto wide = tf::make_geodesic_distances<double>(sphere.polygons(),
                                                        std::vector<int>{7});
  const auto within = tf::make_geodesic_distances<double>(
      sphere.polygons(), std::vector<int>{7}, 1.0f);
  static_assert(std::is_same_v<std::decay_t<decltype(own[0])>, float>,
                "a float mesh states float distances");
  static_assert(std::is_same_v<std::decay_t<decltype(wide[0])>, double>,
                "a stated double is double distances");
  static_assert(std::is_same_v<std::decay_t<decltype(within[0])>, double>,
                "a stated double is double distances within a radius");
  for (std::size_t k = 0; k < sphere.points().size(); ++k) {
    REQUIRE(own[k] == float(wide[k]));
    if (wide[k] > 1.0)
      REQUIRE(std::isinf(within[k]));
  }
}

TEST_CASE("geodesic_distances: a strip of needles unfolds nothing, so no "
          "point falls below the straight line and the front does not "
          "cascade",
          "[geometry][geodesic]") {
  const int n = 10;
  const auto strip = geodesic_needle_strip(n, 1e-5, 8);
  const int seed = n / 2;
  tbb::task_arena one(1);
  tf::geometry::geodesic_front_stats stats;
  const auto distances = one.execute([&] {
    return tf::geometry::geodesic_distances_within<double>(
        strip.polygons(), std::vector<int>{seed},
        std::numeric_limits<double>::infinity(), stats);
  });
  const auto paths = geodesic_edge_paths(strip, {seed});
  const std::size_t n_points = strip.points().size();
  for (std::size_t k = 0; k < n_points; ++k) {
    const double line =
        geodesic_euclid(strip.points()[k], strip.points()[std::size_t(seed)]);
    CAPTURE(k, distances[k], line, paths[k]);
    REQUIRE(distances[k] >= line * (1.0 - 1e-9));
    REQUIRE(distances[k] <= paths[k] * (1.0 + 1e-9));
  }
  CAPTURE(stats.pops, n_points);
  REQUIRE(stats.pops < 3 * n_points);
}
