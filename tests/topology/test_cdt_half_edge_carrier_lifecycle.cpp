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
#include <trueform/topology/cdt/clear_constrained_delaunay.hpp>
#include <trueform/topology/cdt/compact_half_edges.hpp>
#include <trueform/topology/cdt/constrained_delaunay_owner.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/retained_half_edges.hpp>

#include <cstdint>
#include <utility>

namespace {

using compact_t = tf::topology::cdt::compact_half_edges<int>;
using retained_t = tf::topology::cdt::retained_half_edges<int>;
using owner_t = tf::topology::cdt::constrained_delaunay_owner<
    int, float, tf::exact::int32,
    tf::topology::cdt::serial_delaunay_execution_policy>;

auto append_compact_pair(compact_t &edges, int first_vertex) -> void {
  const auto first = edges.append_pair();
  auto first_edge = edges[first];
  first_edge.vertex = first_vertex;
  first_edge.prev = int(first);
  first_edge.next = int(first);
  auto second_edge = edges[first + 1];
  second_edge.vertex = first_vertex + 1;
  second_edge.prev = int(first + 1);
  second_edge.next = int(first + 1);
}

auto append_retained_pair(retained_t &edges, int first_vertex) -> void {
  const retained_t::value_type first{first_vertex, 0,    0,
                                     false,        true, std::uint8_t(1)};
  const retained_t::value_type second{first_vertex + 1, 1, 1, true, false,
                                      std::uint8_t(2)};
  edges.push_back(first);
  edges.push_back(second);
}

} // namespace

TEST_CASE("compact half-edge carriers stay reusable after a move",
          "[constrained_delaunay]") {
  compact_t source;
  append_compact_pair(source, 10);

  compact_t copied(source);
  REQUIRE(copied.size() == 2);
  REQUIRE(copied[0].vertex == 10);

  compact_t moved(std::move(source));
  REQUIRE(moved.size() == 2);
  REQUIRE(moved[1].vertex == 11);
  REQUIRE(source.size() == 0);
  append_compact_pair(source, 20);
  REQUIRE(source[0].vertex == 20);

  compact_t assigned;
  append_compact_pair(assigned, 30);
  assigned = std::move(moved);
  REQUIRE(assigned.size() == 2);
  REQUIRE(assigned[0].vertex == 10);
  REQUIRE(moved.size() == 0);
  append_compact_pair(moved, 40);
  REQUIRE(moved[1].vertex == 41);
}

TEST_CASE("retained half-edge carriers stay reusable after a move",
          "[constrained_delaunay]") {
  retained_t source;
  append_retained_pair(source, 10);

  retained_t copied;
  copied = source;
  REQUIRE(copied.size() == 2);
  REQUIRE(copied[0].constrained == std::uint8_t(1));

  retained_t moved(std::move(source));
  REQUIRE(moved.size() == 2);
  REQUIRE(moved[1].boundary == std::uint8_t(1));
  REQUIRE(source.size() == 0);
  append_retained_pair(source, 20);
  REQUIRE(source[0].vertex == 20);

  retained_t assigned;
  append_retained_pair(assigned, 30);
  assigned = std::move(moved);
  REQUIRE(assigned.size() == 2);
  REQUIRE(assigned[1].constrained == std::uint8_t(2));
  REQUIRE(moved.size() == 0);
  append_retained_pair(moved, 40);
  REQUIRE(moved[1].vertex == 41);
}

TEST_CASE("clearing a constrained owner resets its build scalars",
          "[constrained_delaunay]") {
  owner_t owner;
  owner._orientation_fits_t1 = true;
  owner._incircle_inner_fits_t1 = true;
  owner._allow_crossing_splits = true;
  owner._track_constraint_provenance = true;
  owner._last_edge = 4;
  owner._region_mode = tf::cdt_region_mode::components;
  owner._locate_hint = 5;

  tf::topology::cdt::clear_constrained_delaunay(owner);

  REQUIRE(!owner._orientation_fits_t1);
  REQUIRE(!owner._incircle_inner_fits_t1);
  REQUIRE(!owner._allow_crossing_splits);
  REQUIRE(!owner._track_constraint_provenance);
  REQUIRE(owner._last_edge == owner_t::none);
  REQUIRE(owner._region_mode == tf::cdt_region_mode::nesting);
  REQUIRE(owner._locate_hint == owner_t::none);
}
