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
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>

#include <cstddef>
#include <cstdint>

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

template <typename Int> auto rim_law_holds(bool stacked) -> void {
  auto fixture = tf::test::make_plane_rim_world<Int>(stacked);
  const auto get_original = [&](int, rim_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::arrangement::plane_arrangement<rim_index_t, Int> product;
  product.record_triangle_arrangement();
  product.build(fixture.input, rim_index_t(0), get_original, get_original,
                fixture.vertex_offsets);
  REQUIRE(product.failed().size() == 0);
  // the carried point is stated, so the face refused once and recovered
  CHECK(product.census().refusals >= 1);
  // the two cut pieces are the face's own sides now: each is named by the
  // face's triangle along it
  CHECK(rim_cut_slots(fixture.input, product) >= 2);
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
