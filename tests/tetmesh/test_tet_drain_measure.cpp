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
#include <trueform/tetmesh/mesh/run_tet_census_rounds.hpp>
#include <trueform/tetmesh/mesh/tet_drain_measure.hpp>

#include <cstddef>
#include <utility>

namespace {

using tf::tetmesh::tet_drain_end;

/// The drains a loop over `measure` runs before it ends, each drain handed
/// `crossings(at)` and minting `mints(at)` on `sites(at)` standing, at most
/// `cap` of them; the end it reached, `none` where the cap cut it.
template <typename Crossings, typename Mints, typename Sites>
auto dm_drains(tf::tetmesh::tet_drain_measure measure,
               const Crossings &crossings, const Mints &mints,
               const Sites &sites, std::size_t cap)
    -> std::pair<std::size_t, tet_drain_end> {
  for (std::size_t at = 0; at < cap; ++at) {
    auto end = measure.open(crossings(at));
    if (end == tet_drain_end::none)
      end = measure.cut(0, 1);
    if (end == tet_drain_end::none)
      end = measure.mint(sites(at), mints(at));
    if (end == tet_drain_end::none)
      end = measure.seat(0);
    if (end != tet_drain_end::none)
      return {at, end};
  }
  return {cap, tet_drain_end::none};
}

} // namespace

TEST_CASE("tet drain measure: a drain that neither mints nor reduces the "
          "crossings it was handed refuses as stalled",
          "[tetmesh][liveness]") {
  tf::tetmesh::tet_drain_measure measure;
  measure.budget = 100;
  // the same five crossings, nothing minted: the second drain stalls
  const auto [drains, end] = dm_drains(
      measure, [](std::size_t) { return std::size_t(5); },
      [](std::size_t) { return std::size_t(0); },
      [](std::size_t) { return std::size_t(10); }, 1000);
  CHECK(end == tet_drain_end::stalled);
  CHECK(drains == 1);
}

TEST_CASE("tet drain measure: a drain that reduces the crossings goes on "
          "without minting",
          "[tetmesh][liveness]") {
  tf::tetmesh::tet_drain_measure measure;
  measure.budget = 100;
  // 5, 4, 3, 2, 1, 0 crossings: every drain reduces, the sixth converges
  const auto [drains, end] = dm_drains(
      measure, [](std::size_t at) { return std::size_t(5 - at); },
      [](std::size_t) { return std::size_t(0); },
      [](std::size_t) { return std::size_t(10); }, 1000);
  CHECK(end == tet_drain_end::converged);
  CHECK(drains == 5);
}

TEST_CASE("tet drain measure: minting drains consume the one site budget",
          "[tetmesh][liveness]") {
  tf::tetmesh::tet_drain_measure measure;
  measure.budget = 20;
  // every drain mints one site on the same crossings: the budget ends them
  const auto [drains, end] = dm_drains(
      measure, [](std::size_t) { return std::size_t(5); },
      [](std::size_t) { return std::size_t(1); },
      [](std::size_t at) { return std::size_t(10 + at); }, 1000);
  CHECK(end == tet_drain_end::budget);
  CHECK(drains == 10);
}

TEST_CASE("tet drain measure: every refusal of a drain is its verdict",
          "[tetmesh][liveness]") {
  tf::tetmesh::tet_drain_measure measure;
  measure.budget = 10;
  CHECK(measure.open(0) == tet_drain_end::converged);
  CHECK(measure.open(3) == tet_drain_end::none);
  // a wave that left a carrier refusing or a T-junction
  CHECK(measure.cut(2, 4) == tet_drain_end::failed);
  // a drain that retired no facet
  CHECK(measure.cut(0, 0) == tet_drain_end::unretired);
  // mints the budget cannot hold
  CHECK(measure.mint(8, 3) == tet_drain_end::budget);
  CHECK(measure.mint(12, 0) == tet_drain_end::budget);
  CHECK(measure.mint(8, 2) == tet_drain_end::none);
  // a mint the complex withdrew
  CHECK(measure.seat(1) == tet_drain_end::withdrawn);
  CHECK(measure.seat(0) == tet_drain_end::none);
}

TEST_CASE("tet census rounds: a census the cap cuts short counts what its "
          "last round moved",
          "[tetmesh][liveness]") {
  std::size_t refused = 0, rounds = 0;
  // every round collapses: the cap ends it, its last round's faces counted
  CHECK(tf::tetmesh::run_tet_census_rounds(
      8,
      [&rounds]() -> tf::tetmesh::tet_census_round {
        ++rounds;
        return {1, 3, true};
      },
      refused));
  CHECK(rounds == 8);
  CHECK(refused == 3);
  // a round that collapses nothing ends it with nothing refused
  refused = 0;
  rounds = 0;
  CHECK(tf::tetmesh::run_tet_census_rounds(
      8,
      [&rounds]() -> tf::tetmesh::tet_census_round {
        ++rounds;
        return {rounds < 3 ? std::size_t(1) : std::size_t(0), 3, true};
      },
      refused));
  CHECK(rounds == 3);
  CHECK(refused == 0);
  // a round whose recovery runs out of capacity ends it as false
  CHECK_FALSE(tf::tetmesh::run_tet_census_rounds(
      8, []() -> tf::tetmesh::tet_census_round { return {1, 3, false}; },
      refused));
}
