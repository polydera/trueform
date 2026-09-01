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
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

using owner_api_index_t = std::int32_t;
using Coord = std::int32_t;
using triangulator_type =
    tf::constrained_delaunay_triangulator<owner_api_index_t, Coord,
                                          tf::exact::int32>;

auto make_owner_api_points() -> tf::points_buffer<Coord, 2> {
  tf::points_buffer<Coord, 2> points;
  points.push_back(tf::make_point(Coord(0), Coord(0)));
  points.push_back(tf::make_point(Coord(10), Coord(0)));
  points.push_back(tf::make_point(Coord(10), Coord(10)));
  points.push_back(tf::make_point(Coord(0), Coord(10)));
  return points;
}

constexpr std::array<std::array<owner_api_index_t, 2>, 4> constraints{
    std::array<owner_api_index_t, 2>{0, 1},
    std::array<owner_api_index_t, 2>{1, 2},
    std::array<owner_api_index_t, 2>{2, 3},
    std::array<owner_api_index_t, 2>{3, 0}};

/// Every retained owner names its input constraint and spans it whole, with
/// the parameters reading forward or backward according to the face
/// orientation of that edge.
auto has_owner(const triangulator_type &triangulator) -> bool {
  const auto faces = triangulator.faces();
  const auto &input_to_output = triangulator.index_map().f();
  const auto whole = triangulator_type::param_t(1)
                     << triangulator_type::crossing_param_bits();
  std::size_t owner_count = 0;
  for (std::size_t triangle = 0; triangle < faces.size(); ++triangle) {
    const auto owners =
        triangulator.face_constraint_owners(owner_api_index_t(triangle));
    for (std::size_t edge = 0; edge < 3; ++edge) {
      const auto owner = owners[edge];
      if (owner.input_id == triangulator_type::k_none)
        continue;
      ++owner_count;
      const auto input_edge = constraints[std::size_t(owner.input_id)];
      const owner_api_index_t first =
          input_to_output[std::size_t(input_edge[0])];
      const owner_api_index_t second =
          input_to_output[std::size_t(input_edge[1])];
      const owner_api_index_t face_first = faces[triangle][edge];
      const owner_api_index_t face_second = faces[triangle][(edge + 1) % 3];
      if (face_first == first && face_second == second) {
        REQUIRE(owner.t0 == 0);
        REQUIRE(owner.t1 == whole);
      } else {
        REQUIRE(face_first == second);
        REQUIRE(face_second == first);
        REQUIRE(owner.t0 == whole);
        REQUIRE(owner.t1 == 0);
      }
    }
  }
  return owner_count == constraints.size();
}

} // namespace

TEST_CASE("CDT no-split nesting builds retain no constraint owners",
          "[constrained_delaunay]") {
  const auto points = make_owner_api_points();
  const auto edges = tf::make_edges(constraints);

  triangulator_type triangulator;
  REQUIRE(triangulator.build(points.points(), edges, false));
  REQUIRE(!has_owner(triangulator));
}

TEST_CASE("CDT always_track_constraint_owners survives clear and rebuild",
          "[constrained_delaunay]") {
  const auto points = make_owner_api_points();
  const auto edges = tf::make_edges(constraints);

  triangulator_type triangulator;
  triangulator.always_track_constraint_owners();
  REQUIRE(triangulator.build(points.points(), edges, false));
  REQUIRE(has_owner(triangulator));

  triangulator.clear();
  REQUIRE(triangulator.build(points.points(), edges, false));
  REQUIRE(has_owner(triangulator));
}
