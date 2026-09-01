/*
 * Copyright (c) 2025 XLAB
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
#include <tbb/global_control.h>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/delaunay_vertex_policy.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>
#include <trueform/topology/unconstrained_delaunay_triangulator.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using index_type = std::int32_t;
using coordinate_type = std::int32_t;
using vertex_policy = tf::topology::cdt::original_input_vertex_policy;
using serial_policy = tf::topology::cdt::serial_delaunay_execution_policy;
using parallel_policy = tf::topology::cdt::parallel_delaunay_execution_policy;

template <typename Faces>
auto canonical_faces(const Faces &faces)
    -> std::vector<std::array<index_type, 3>> {
  std::vector<std::array<index_type, 3>> result;
  result.reserve(faces.size());
  for (const auto face : faces) {
    std::array<index_type, 3> canonical{face[0], face[1], face[2]};
    std::sort(canonical.begin(), canonical.end());
    result.push_back(canonical);
  }
  std::sort(result.begin(), result.end());
  return result;
}

template <typename ExecutionPolicy>
auto triangulate(const tf::points_buffer<coordinate_type, 2> &points)
    -> std::vector<std::array<index_type, 3>> {
  tf::unconstrained_delaunay_triangulator<index_type, coordinate_type,
                                          tf::exact::int32, vertex_policy,
                                          ExecutionPolicy>
      triangulator;
  REQUIRE(triangulator.build(points.points()));
  return canonical_faces(triangulator.faces());
}

template <typename ExecutionPolicy>
auto triangulate_with_constraints(
    const tf::points_buffer<coordinate_type, 2> &points)
    -> std::vector<std::array<index_type, 3>> {
  const std::array<std::array<index_type, 2>, 2> constraint_data{
      std::array<index_type, 2>{0, 1}, std::array<index_type, 2>{2, 3}};
  tf::constrained_delaunay_triangulator<index_type, coordinate_type,
                                        tf::exact::int32, ExecutionPolicy>
      triangulator;
  REQUIRE(triangulator.build_triangulation(
      points.points(), tf::make_edges(constraint_data), false));
  return canonical_faces(triangulator.faces());
}

auto make_sites(std::size_t n_sites) -> tf::points_buffer<coordinate_type, 2> {
  tf::points_buffer<coordinate_type, 2> points;
  points.reserve(n_sites);
  std::uint64_t state = 0x6a09e667f3bcc909ULL;
  for (std::size_t i = 0; i < n_sites; ++i) {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    const auto x = coordinate_type(-900000000 + std::int64_t(i) * 30000);
    const auto y = coordinate_type(state % 1800000001ULL) - 900000000;
    points.push_back(tf::make_point(x, y));
  }
  return points;
}

} // namespace

TEST_CASE("Delaunay topology is identical under serial and parallel policies",
          "[constrained_delaunay]") {
  const auto points = make_sites(60000);

  const auto serial = triangulate<serial_policy>(points);
  const tbb::global_control workers(
      tbb::global_control::max_allowed_parallelism, 4);
  const auto parallel = triangulate<parallel_policy>(points);
  const auto repeated = triangulate<parallel_policy>(points);
  REQUIRE(serial == parallel);
  REQUIRE(parallel == repeated);

  const auto constrained_serial =
      triangulate_with_constraints<serial_policy>(points);
  const auto constrained_parallel =
      triangulate_with_constraints<parallel_policy>(points);
  REQUIRE(constrained_serial == constrained_parallel);
}
