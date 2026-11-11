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
#include "uniform_draw.hpp"
#include "worker_ladder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/dt3/tet_execution_tuning.hpp>
#include <trueform/topology/delaunay_tetrahedralizer.hpp>

#include <cstddef>
#include <random>
#include <tbb/task_arena.h>
#include <vector>

namespace {

using dt3h_i32 = tf::exact::int32;

auto dt3h_random_coordinates(unsigned seed, std::size_t n)
    -> std::vector<dt3h_i32> {
  std::mt19937 rng{seed};
  std::vector<dt3h_i32> flat(3 * n);
  for (auto &value : flat)
    value = tf::test::uniform_int(rng, dt3h_i32(0), dt3h_i32((1 << 20) - 1));
  return flat;
}

/// Every row an owner publishes — its finite cells, its hull cells, the
/// links of both and their states — and the row a walk starts from.
struct dt3h_rows {
  std::vector<int> corners;
  std::vector<int> neighbors;
  std::vector<int> states;
  int hint;
};

auto operator==(const dt3h_rows &a, const dt3h_rows &b) -> bool {
  return a.corners == b.corners && a.neighbors == b.neighbors &&
         a.states == b.states && a.hint == b.hint;
}

template <typename Builder>
auto dt3h_published_rows(const Builder &builder) -> dt3h_rows {
  const auto &owner = builder.owner();
  dt3h_rows rows;
  for (auto corner : owner._corners.data_buffer())
    rows.corners.push_back(int(corner));
  for (auto neighbor : owner._neighbors.data_buffer())
    rows.neighbors.push_back(int(neighbor));
  for (auto state : owner._states)
    rows.states.push_back(int(state));
  rows.hint = int(owner._hint);
  return rows;
}

} // namespace

TEST_CASE("delaunay tets: every published row is one order at any worker "
          "count",
          "[topology][delaunay3][parallel]") {
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  using site_type = tf::exact::vertex<int, dt3h_i32>;
  const std::size_t workers = tf::test::machine_workers();
  if (workers < Tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
  const std::size_t n = 2 * Tuning::parallel_sites;
  const std::size_t appended = 2 * Tuning::parallel_append_sites;
  const auto flat = dt3h_random_coordinates(11, n + appended);
  const auto points =
      tf::make_points<3>(tf::make_range(flat.data(), flat.data() + 3 * n));
  std::vector<site_type> sites;
  for (std::size_t i = n; i < n + appended; ++i)
    sites.push_back({int(i), {flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]}});

  tf::delaunay_tetrahedralizer<
      int, dt3h_i32, dt3h_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>
      serial;
  REQUIRE(serial.build(points));
  const auto built = dt3h_published_rows(serial);
  REQUIRE(serial.append_sites(tf::make_range(sites)));
  const auto grown = dt3h_published_rows(serial);
  CHECK(built.hint == 0);
  CHECK(grown.hint == 0);

  for (const std::size_t width : tf::test::worker_ladder({1, workers})) {
    CAPTURE(width);
    tbb::task_arena arena(static_cast<int>(width));
    tf::delaunay_tetrahedralizer<int, dt3h_i32> parallel;
    REQUIRE(arena.execute([&] { return parallel.build(points); }));
    CHECK((parallel.stats().parallel_insertions > 0) == (width > 1));
    CHECK(dt3h_published_rows(parallel) == built);
    REQUIRE(arena.execute(
        [&] { return parallel.append_sites(tf::make_range(sites)); }));
    CHECK(dt3h_published_rows(parallel) == grown);
  }
}
