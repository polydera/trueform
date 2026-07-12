/**
 * @file test_clean_loop.cpp
 * @brief Tests for tf::intersect::graph::clean_loop / clean_loops.
 *
 * Base loops must carry no consecutive duplicates and no antennas
 * (X, A, X spikes), including across the wrap-around seam.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/intersect/graph/clean_loop.hpp>

#include <vector>

using Index = int;
using vertex_t = tf::intersect::graph::vertex<Index>;
namespace vsrc = tf::intersect::graph;

namespace {

auto orig(Index id) -> vertex_t {
  return {vsrc::vertex_source::original, id, {0, tf::topo_type::vertex}};
}
auto created(Index id) -> vertex_t {
  return {vsrc::vertex_source::created, id, {0, tf::topo_type::edge}};
}

auto clean(const std::vector<vertex_t> &dirty) -> std::vector<Index> {
  tf::buffer<vertex_t> in;
  for (auto &v : dirty)
    in.push_back(v);
  tf::buffer<vertex_t> out;
  tf::intersect::graph::clean_loop<Index>(tf::make_range(in), out);
  std::vector<Index> ids;
  for (auto &v : out)
    ids.push_back(v.source == vsrc::vertex_source::created ? ~v.id : v.id);
  return ids;
}

// created ids print as ~id (bitwise not) to distinguish sources
auto ids(std::initializer_list<Index> l) -> std::vector<Index> { return l; }

} // namespace

TEST_CASE("clean_loop: already clean loop is unchanged",
          "[intersect][clean_loop]") {
  REQUIRE(clean({orig(0), orig(1), orig(2)}) == ids({0, 1, 2}));
  REQUIRE(clean({orig(0), created(5), orig(1), orig(2)}) ==
          ids({0, ~5, 1, 2}));
}

TEST_CASE("clean_loop: consecutive duplicates removed",
          "[intersect][clean_loop]") {
  SECTION("single dup mid-loop") {
    REQUIRE(clean({orig(0), orig(1), orig(1), orig(2)}) == ids({0, 1, 2}));
  }
  SECTION("run of dups") {
    REQUIRE(clean({orig(0), orig(1), orig(1), orig(1), orig(2)}) ==
            ids({0, 1, 2}));
  }
  SECTION("dup across the wrap seam") {
    REQUIRE(clean({orig(0), orig(1), orig(2), orig(0)}) == ids({0, 1, 2}));
  }
  SECTION("same id, different source is NOT a duplicate") {
    REQUIRE(clean({orig(0), orig(1), created(1), orig(2)}) ==
            ids({0, 1, ~1, 2}));
  }
}

TEST_CASE("clean_loop: antennas snipped", "[intersect][clean_loop]") {
  SECTION("mid-loop spike (X, A, X) -> (X)") {
    REQUIRE(clean({orig(0), orig(1), orig(9), orig(1), orig(2)}) ==
            ids({0, 1, 2}));
  }
  SECTION("chained spike collapses inward (X, A, B, A, X) -> (X)") {
    REQUIRE(clean({orig(0), orig(1), orig(8), orig(9), orig(8), orig(1),
                   orig(2)}) == ids({0, 1, 2}));
  }
  SECTION("spike across the seam: back equals front+1") {
    // loop (A, X, B, C, X): wrap makes (X, A, X) at the seam -> A gone
    REQUIRE(clean({orig(1), orig(0), orig(2), orig(3), orig(0)}) ==
            ids({0, 2, 3}));
  }
  SECTION("spike across the seam: back-1 equals front") {
    // loop (X, B, C, X, A): (X, A, X) wrapping the other way
    REQUIRE(clean({orig(0), orig(2), orig(3), orig(0), orig(9)}) ==
            ids({0, 2, 3}));
  }
  SECTION("created-point antenna on an edge") {
    REQUIRE(clean({orig(0), created(7), orig(1), created(7), orig(2),
                   orig(3)}) == ids({0, ~7, 2, 3}));
  }
}

TEST_CASE("clean_loop: combined dups and antennas",
          "[intersect][clean_loop]") {
  REQUIRE(clean({orig(0), orig(0), orig(1), orig(9), orig(9), orig(1),
                 orig(2), orig(2)}) == ids({0, 1, 2}));
}

TEST_CASE("clean_loop: degenerate inputs pass through the small-loop path",
          "[intersect][clean_loop]") {
  REQUIRE(clean({orig(0), orig(1)}) == ids({0, 1}));
  REQUIRE(clean({orig(0)}) == ids({0}));
  REQUIRE(clean({}) == ids({}));
}

TEST_CASE("clean_loop: loop that is pure antenna collapses",
          "[intersect][clean_loop]") {
  // (X, A, X) exactly: wrap dup + spike leaves a single vertex
  auto r = clean({orig(0), orig(1), orig(0)});
  REQUIRE(r.size() <= 1);
}

TEST_CASE("clean_loop: idempotent", "[intersect][clean_loop]") {
  std::vector<vertex_t> dirty{orig(0), orig(1), orig(9), orig(1),
                              created(4), orig(2), orig(2), orig(0)};
  tf::buffer<vertex_t> in;
  for (auto &v : dirty)
    in.push_back(v);
  tf::buffer<vertex_t> once;
  tf::intersect::graph::clean_loop<Index>(tf::make_range(in), once);
  tf::buffer<vertex_t> twice;
  tf::intersect::graph::clean_loop<Index>(tf::make_range(once), twice);
  REQUIRE(once.size() == twice.size());
  for (std::size_t i = 0; i < once.size(); ++i)
    REQUIRE((once[i].id == twice[i].id &&
             once[i].source == twice[i].source));
}

TEST_CASE("clean_loops: batch over offset blocks", "[intersect][clean_loop]") {
  tf::offset_block_buffer<Index, vertex_t> loops;
  loops.push_back({orig(0), orig(1), orig(9), orig(1), orig(2)}); // antenna
  loops.push_back({orig(3), orig(3), orig(4), orig(5)});          // dup
  loops.push_back({orig(6), orig(7), orig(8)});                   // clean
  tf::intersect::graph::clean_loops(loops);
  REQUIRE(loops.size() == 3);
  REQUIRE(loops[0].size() == 3);
  REQUIRE(loops[1].size() == 3);
  REQUIRE(loops[2].size() == 3);
  REQUIRE(loops[0][0].id == 0);
  REQUIRE(loops[0][2].id == 2);
  REQUIRE(loops[1][0].id == 3);
}
