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
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/delaunay_tetrahedralizer.hpp>
#include <trueform/topology/tetrahedralization_refusal.hpp>

namespace {

using dt3e_i32 = tf::exact::int32;

/// A build over no site refuses as rank-deficient, publishes no cell, and
/// leaves the builder ready for the build that follows.
template <typename Builder> auto dt3e_check_no_site(Builder &builder) -> void {
  tf::points_buffer<dt3e_i32, 3> points;
  CHECK_FALSE(builder.build(points.points()));
  CHECK(builder.refusal() == tf::tetrahedralization_refusal::rank_deficient);
  CHECK(builder.n_tets() == 0);
  CHECK(builder.n_sites() == 0);
  CHECK_FALSE(builder.is_valid());
  points.emplace_back(0, 0, 0);
  points.emplace_back(4, 0, 0);
  points.emplace_back(0, 4, 0);
  points.emplace_back(0, 0, 4);
  points.emplace_back(1, 1, 1);
  REQUIRE(builder.build(points.points()));
  CHECK(builder.refusal() == tf::tetrahedralization_refusal::none);
  CHECK(builder.n_tets() == 4);
  CHECK(builder.is_valid());
}

} // namespace

TEST_CASE("delaunay tets: a build over no site is refused as rank-deficient",
          "[topology][delaunay3]") {
  tf::delaunay_tetrahedralizer<int, dt3e_i32> parallel;
  dt3e_check_no_site(parallel);
  tf::delaunay_tetrahedralizer<
      int, dt3e_i32, dt3e_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>
      serial;
  dt3e_check_no_site(serial);
}
