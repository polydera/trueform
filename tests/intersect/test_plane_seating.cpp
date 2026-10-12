/**
 * @file test_plane_seating.cpp
 * @brief One seating law for every tier that admits faces
 *
 * An entrant names the plane it stands on: a name a standing plane holds
 * seats it on that plane, equal names among the rest found one new plane
 * together, and a void name is a plane of its own. The new planes follow the
 * standing ones in the order of their smallest entrant. The local
 * arrangement's entrance and the plane arrangement's wave entrance both seat
 * through this one law, so two coplanar entrants on one side of a shared edge
 * are one plane identity, never two.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/exact/canonical_plane.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/graph/seat_named_planes.hpp>

#include <vector>

namespace {

using seating_int_t = tf::exact::int32;
using seating_name_t = tf::exact::canonical_plane<seating_int_t>;
using seating_t2 = tf::exact::meta<seating_int_t>::T2;

auto seating_name(int a, int b, int c, int d) -> seating_name_t {
  return {seating_t2(a), seating_t2(b), seating_t2(c), seating_t2(d)};
}

} // namespace

TEST_CASE("plane seating: coplanar entrants are one plane identity",
          "[intersect][planes][seating]") {
  const auto standing_a = seating_name(0, 0, 1, 0);
  const auto standing_b = seating_name(1, 0, 0, 5);
  const auto fresh_c = seating_name(0, 1, 0, 7);
  const auto fresh_d = seating_name(1, 1, 0, 2);
  const auto void_name = seating_name(0, 0, 0, 0);
  const std::vector<seating_name_t> standing{standing_a, standing_b};

  // two coplanar entrants (0, 2), one on a standing plane (1), a collinear
  // face (3) and a lone new plane (4)
  tf::buffer<seating_name_t> names;
  for (const auto &name :
       {fresh_c, standing_a, fresh_c, void_name, fresh_d})
    names.push_back(name);

  tf::buffer<int> plane_of;
  tf::buffer<int> new_plane_first;
  const auto n_new = tf::intersect::graph::seat_named_planes<int, seating_int_t>(
      names, standing, int(standing.size()), plane_of, new_plane_first);

  REQUIRE(n_new == 3);
  // the coplanar pair founds one plane, numbered by its smallest entrant
  CHECK(plane_of[0] == 2);
  CHECK(plane_of[2] == 2);
  // a name a standing plane holds joins it
  CHECK(plane_of[1] == 0);
  // a void name stands alone, and the rest follow by smallest entrant
  CHECK(plane_of[3] == 3);
  CHECK(plane_of[4] == 4);
  CHECK(new_plane_first[0] == 0);
  CHECK(new_plane_first[1] == 3);
  CHECK(new_plane_first[2] == 4);
}

TEST_CASE("plane seating: void names never share a plane",
          "[intersect][planes][seating]") {
  const auto void_name = seating_name(0, 0, 0, 0);
  const std::vector<seating_name_t> standing{void_name};
  tf::buffer<seating_name_t> names;
  names.push_back(void_name);
  names.push_back(void_name);
  tf::buffer<int> plane_of;
  tf::buffer<int> new_plane_first;
  const auto n_new = tf::intersect::graph::seat_named_planes<int, seating_int_t>(
      names, standing, int(standing.size()), plane_of, new_plane_first);
  REQUIRE(n_new == 2);
  CHECK(plane_of[0] == 1);
  CHECK(plane_of[1] == 2);
}
