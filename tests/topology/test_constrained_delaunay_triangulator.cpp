#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <array>
#include <initializer_list>

namespace {

using Index = int;
using Int = tf::exact::int32;
using Cdt = tf::constrained_delaunay_triangulator<Index, Int, Int>;

auto make_points(std::initializer_list<std::array<Int, 2>> values)
    -> tf::points_buffer<Int, 2> {
  tf::points_buffer<Int, 2> points;
  for (const auto &value : values)
    points.push_back(tf::point<Int, 2>{value[0], value[1]});
  return points;
}

} // namespace

TEST_CASE("CDT crossings use welded output point IDs",
          "[constrained_delaunay]") {
  auto points =
      make_points({{0, 0}, {4, 0}, {5, 1}, {2, 0}, {5, -1}});
  std::array<std::array<Index, 2>, 3> edge_data{
      std::array<Index, 2>{0, 1}, std::array<Index, 2>{2, 3},
      std::array<Index, 2>{3, 4}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  const Index junction = cdt.index_map().f()[3];
  const auto &crossings = cdt.constraint_crossings();
  REQUIRE(crossings.size() == cdt.points().size());
  REQUIRE(crossings[junction].size() == 1);
  CHECK(crossings[junction][0] == 0);
  CHECK(cdt.index_map().kept_ids()[junction] == 3);

  Index callback_count = 0;
  cdt.for_each_constraint_crossing(
      [&](Index point, const auto &constraints) {
        ++callback_count;
        CHECK(point == junction);
        REQUIRE(constraints.size() == 1);
        CHECK(constraints[0] == 0);
      });
  CHECK(callback_count == 1);
}

TEST_CASE("CDT crossings retain synthetic point constraint incidences",
          "[constrained_delaunay]") {
  auto points = make_points({{-2, 0}, {2, 0}, {0, -2}, {0, 2}});
  std::array<std::array<Index, 2>, 3> edge_data{
      std::array<Index, 2>{0, 0}, std::array<Index, 2>{0, 1},
      std::array<Index, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  const auto &crossings = cdt.constraint_crossings();
  Index crossing_point = Index(-1);
  for (Index point = 0; point < static_cast<Index>(crossings.size()); ++point) {
    if (crossings[point].size() == 0)
      continue;
    REQUIRE(crossing_point == Index(-1));
    crossing_point = point;
  }

  REQUIRE(crossing_point != Index(-1));
  REQUIRE(crossings[crossing_point].size() == 2);
  CHECK(crossings[crossing_point][0] == 1);
  CHECK(crossings[crossing_point][1] == 2);
  CHECK(cdt.index_map().kept_ids()[crossing_point] ==
        static_cast<Index>(points.size()));
}
