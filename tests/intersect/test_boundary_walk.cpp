/**
 * @file test_boundary_walk.cpp
 * @brief Tests for tf::intersect::graph::emit_boundary_sub_edges — the
 * walk must never emit through an original corner (a snipped on-edge
 * occurrence with a surviving far pinch occurrence would otherwise put
 * mesh-vertex ids in intersection-point fields).
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/range.hpp>
#include <trueform/intersect/graph/edges.hpp>

using Index = int;
using vertex_t = tf::intersect::graph::vertex<Index>;
namespace ig = tf::intersect::graph;

namespace {

auto orig(Index id) -> vertex_t {
  return {ig::vertex_source::original, id, {0, tf::topo_type::vertex}};
}
auto created(Index id) -> vertex_t {
  return {ig::vertex_source::created, id, {0, tf::topo_type::edge}};
}

auto stamp = [](Index, Index, std::size_t) -> std::array<std::int16_t, 2> {
  return {std::int16_t(-1), std::int16_t(-1)};
};

} // namespace

TEST_CASE("emit_boundary_sub_edges: created chain on one edge walks",
          "[intersect][edges]") {
  tf::buffer<vertex_t> loop;
  for (auto v : {orig(0), created(10), created(11), created(12), orig(1)})
    loop.push_back(v);
  tf::buffer<ig::edge<Index>> buf;
  bool ok = ig::emit_boundary_sub_edges<Index>(tf::make_range(loop), 10, 12, 0,
                                               0, 0, 1, buf, stamp);
  REQUIRE(ok);
  REQUIRE(buf.size() == 2);
  REQUIRE(buf[0].point_0 == 10);
  REQUIRE(buf[0].point_1 == 11);
  REQUIRE(buf[1].point_0 == 11);
  REQUIRE(buf[1].point_1 == 12);
}

TEST_CASE("emit_boundary_sub_edges: far pinch occurrence must not walk "
          "through original corners",
          "[intersect][edges]") {
  // end_id 20 exists only PAST an original corner (its on-edge
  // occurrence was snipped): the walk must roll back and report false
  // so the caller degrades to a chord.
  tf::buffer<vertex_t> loop;
  for (auto v :
       {orig(0), created(10), created(11), orig(1), created(20), orig(2)})
    loop.push_back(v);
  tf::buffer<ig::edge<Index>> buf;
  bool ok = ig::emit_boundary_sub_edges<Index>(tf::make_range(loop), 10, 20, 0,
                                               0, 0, 1, buf, stamp);
  REQUIRE_FALSE(ok);
  REQUIRE(buf.size() == 0);
}

TEST_CASE("emit_boundary_sub_edges: absent endpoint rolls back",
          "[intersect][edges]") {
  tf::buffer<vertex_t> loop;
  for (auto v : {orig(0), created(10), created(11), orig(1)})
    loop.push_back(v);
  tf::buffer<ig::edge<Index>> buf;
  bool ok = ig::emit_boundary_sub_edges<Index>(tf::make_range(loop), 10, 99, 0,
                                               0, 0, 1, buf, stamp);
  REQUIRE_FALSE(ok);
  REQUIRE(buf.size() == 0);
}
