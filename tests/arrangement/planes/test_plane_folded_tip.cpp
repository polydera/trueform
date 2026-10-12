/**
 * @file test_plane_folded_tip.cpp
 * @brief A cancelled wall still joins the loop it cancels in
 *
 * A face's loop that states one wall both ways cancels that wall's winding,
 * but still runs it twice. When the wall is the only thing joining two parts
 * of the loop — the body of a needle and its tip, cut off by a crossing of
 * the needle's own sides — and the rounding turned the tip inside out, the
 * tip winds against the face. Every triangle must still run the way the loop
 * winds around it, so each wall the loop states is emitted once in its own
 * direction and the cancelled wall not at all.
 *
 * The world is stated by hand, on a lone member.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "plane_arrangement_generators.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace {

using folded_tip_index_t = tf::test::plane_index_t;

template <typename Int> auto folded_tip_balances() -> void {
  auto fixture = tf::test::make_plane_folded_tip_world<Int>();
  const auto get_original = [&](int, folded_tip_index_t point) {
    return fixture.points[std::size_t(point)];
  };
  tf::arrangement::plane_arrangement<folded_tip_index_t, Int> product;
  product.build(fixture.input, folded_tip_index_t(0), get_original,
                get_original, fixture.vertex_offsets);
  REQUIRE(product.failed().size() == 0);

  // the net direction the emitted triangles run each edge
  std::vector<std::array<int, 3>> runs;
  for (const auto &triangle : product.triangles())
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const int from = int(triangle[corner]);
      const int to = int(triangle[(corner + 1) % 3]);
      runs.push_back({std::min(from, to), std::max(from, to),
                      from < to ? 1 : -1});
    }
  std::sort(runs.begin(), runs.end());
  std::vector<std::array<int, 3>> net;
  for (const auto &run : runs) {
    if (!net.empty() && net.back()[0] == run[0] && net.back()[1] == run[1])
      net.back()[2] += run[2];
    else
      net.push_back(run);
  }
  net.erase(std::remove_if(net.begin(), net.end(),
                           [](const std::array<int, 3> &edge) {
                             return edge[2] == 0;
                           }),
            net.end());

  // the loop's walls, each in its own direction: A M, S T, T B, B S, M C, C A
  const std::vector<std::array<int, 3>> walls{
      {0, 2, -1}, {0, 5, 1}, {1, 3, -1}, {1, 4, 1}, {2, 5, -1}, {3, 4, -1}};
  CHECK(net == walls);
}

} // namespace

TEMPLATE_TEST_CASE("plane folded tip: a cancelled wall joins the loop",
                   "[arrangement][planes][winding]", tf::exact::int32,
                   tf::exact::int64) {
  folded_tip_balances<TestType>();
}
