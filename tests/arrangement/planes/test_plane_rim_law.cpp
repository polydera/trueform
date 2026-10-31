/**
 * @file test_plane_rim_law.cpp
 * @brief The rim law: a face's cut never leaves the face
 *
 * A cut ends where it pierces its face, and the pierce is inside the face
 * exactly; the one rounding may still carry it across a side, beyond the
 * pieces that side was split into. The cut then bounds a sliver its face
 * covers on neither side, nothing crosses anything, and no emitted triangle
 * of the face names the cut — so the cut's pages lose the face and two sides
 * of one surface meet across it. The law lands the carried point on the side
 * it crossed: the wall is split at an identity that already stands, the cut's
 * pieces become the face's own sides, and every cut piece is named by a slot
 * of the face that states it.
 *
 * The world is stated by hand, so the point sits one lattice unit outside its
 * side exactly, on a lone member and on a stack.
 *
 * A side the refinement graded rounds each of its points on its own, so the
 * sliver can hold several of the face's corners, each a sub-unit across the
 * cut. Every one of them stands on the cut on the lattice, and all of them
 * land in one round: no cut piece the plane carries is left unnamed by its
 * slots.
 *
 * Refinement plans only for a carrier the round accepts: the rim world's
 * seed builds, so its discovery reaches the acceptance, and the landing the
 * round would refuse on declines the plan instead.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/identity/identity_substrate.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/topology/cdt_refine_config.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using rim_index_t = tf::test::plane_index_t;

/// How many emitted slots name a piece the face states as a cut.
template <typename World, typename Product>
auto rim_cut_slots(const World &world, const Product &product) -> std::size_t {
  std::size_t named = 0;
  for (const auto &slots : product.slot_parents())
    for (const auto ticket : slots) {
      if (ticket < rim_index_t(0))
        continue;
      bool cut = false;
      for (const auto &definition : product.piece_definitions(world, ticket))
        cut = cut || definition.side < 0;
      named += cut ? 1 : 0;
    }
  return named;
}

/// How many cut pieces the plane carries that no emitted slot of it names.
template <typename World, typename Product>
auto rim_unnamed_cuts(const World &world, const Product &product)
    -> std::size_t {
  const auto blocks = product.plane_tickets();
  const auto block = blocks.size() == 0 ? rim_index_t(-1) : blocks[0];
  std::vector<rim_index_t> carried;
  if (block == rim_index_t(-1))
    for (const auto row : world.plane_edges(0))
      carried.push_back(world.edge_defs()[std::size_t(row)].id);
  else
    for (const auto ticket : product.current_plane_tickets(world, block))
      carried.push_back(ticket);
  std::vector<rim_index_t> named;
  const auto range = product.plane_range(0);
  for (auto row = range[0]; row < range[1]; ++row)
    for (const auto ticket : product.slot_parents()[std::size_t(row)])
      if (ticket >= rim_index_t(0))
        named.push_back(ticket);
  std::sort(named.begin(), named.end());
  std::size_t unnamed = 0;
  for (const auto ticket : carried) {
    bool cut = false;
    for (const auto &definition : product.piece_definitions(world, ticket))
      cut = cut || definition.side < 0;
    if (cut && !std::binary_search(named.begin(), named.end(), ticket))
      ++unnamed;
  }
  return unnamed;
}

template <typename Int> auto rim_chain_lands(bool stacked) -> void {
  auto fixture = tf::test::make_plane_rim_chain_world<Int>(stacked);
  const auto get_original = [&](int, rim_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::arrangement::plane_arrangement<rim_index_t, Int> product;
  tf::arrangement::identity_substrate<rim_index_t, Int> identities;
  product.record_triangle_arrangement();
  product.build(fixture.input, identities, rim_index_t(0), get_original,
                get_original, fixture.vertex_offsets);
  REQUIRE(product.failed().size() == 0);
  // both chain corners land in the one refusal
  CHECK(product.census().rim_landings == 2);
  CHECK(product.census().uncovered_cuts == 0);
  CHECK(rim_unnamed_cuts(fixture.input, product) == 0);
  CHECK(rim_cut_slots(fixture.input, product) >= 3);
}

template <typename Int> auto rim_law_holds(bool stacked) -> void {
  auto fixture = tf::test::make_plane_rim_world<Int>(stacked);
  const auto get_original = [&](int, rim_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::arrangement::plane_arrangement<rim_index_t, Int> product;
  tf::arrangement::identity_substrate<rim_index_t, Int> identities;
  product.record_triangle_arrangement();
  product.build(fixture.input, identities, rim_index_t(0), get_original,
                get_original, fixture.vertex_offsets);
  REQUIRE(product.failed().size() == 0);
  // the carried point is stated, so the face refused once and recovered
  CHECK(product.census().refusals >= 1);
  // the two cut pieces are the face's own sides now: each is named by the
  // face's triangle along it
  CHECK(rim_cut_slots(fixture.input, product) >= 2);
}

template <typename Int> auto rim_plan_declined(bool stacked) -> void {
  auto fixture = tf::test::make_plane_rim_world<Int>(stacked);
  const auto get_original = [&](int, rim_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::cdt_refine_config config;
  config.min_quality = 0.30f;
  tf::arrangement::plane_arrangement<rim_index_t, Int> product;
  tf::arrangement::identity_substrate<rim_index_t, Int> identities;
  product.record_triangle_arrangement();
  product.build_refined(fixture.input, identities, rim_index_t(0), get_original,
                        get_original, fixture.vertex_offsets, config);
  REQUIRE(product.failed().size() == 0);
  // the premise: the round lands the rim on this carrier
  CHECK(product.census().rim_landings >= 1);
  // the admission law declined the carrier the round refuses, so no plan
  // stood on it
  CHECK(product.refinement_census().admission_refused == 1);
  CHECK(product.refinement_census().round1_retries == 0);
  CHECK(product.refinement_refused_planes().size() == 1);
}

} // namespace

TEMPLATE_TEST_CASE("plane rim law: a cut end outside its side lands on it",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_law_holds<TestType>(false);
}

TEMPLATE_TEST_CASE("plane rim law: on a stack the cutting member claims it",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_law_holds<TestType>(true);
}

TEMPLATE_TEST_CASE("plane rim law: a graded chain across the cut lands whole",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_chain_lands<TestType>(false);
}

TEMPLATE_TEST_CASE("plane rim law: a graded chain lands on a stack",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_chain_lands<TestType>(true);
}

TEMPLATE_TEST_CASE("plane rim law: refinement plans no carrier it lands on",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_plan_declined<TestType>(false);
}

TEMPLATE_TEST_CASE("plane rim law: refinement plans no stack it lands on",
                   "[arrangement][planes][rim]", tf::exact::int32,
                   tf::exact::int64) {
  rim_plan_declined<TestType>(true);
}
