#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/views/constant.hpp>
#include <trueform/topology/make_cdt.hpp>
#include <trueform/reindex/return_index_map.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>

namespace {

using make_cdt_index_t = int;

auto make_points(std::initializer_list<std::array<float, 2>> values)
    -> tf::points_buffer<float, 2> {
  tf::points_buffer<float, 2> points;
  for (const auto &value : values)
    points.push_back(tf::point<float, 2>{value[0], value[1]});
  return points;
}

template <typename Polygons>
auto has_edge(const Polygons &polys, make_cdt_index_t a, make_cdt_index_t b)
    -> bool {
  for (auto face : polys.faces())
    for (int c = 0; c < 3; ++c) {
      const make_cdt_index_t u = face[std::size_t(c)];
      const make_cdt_index_t v = face[std::size_t((c + 1) % 3)];
      if ((u == a && v == b) || (u == b && v == a))
        return true;
    }
  return false;
}

auto square() -> tf::points_buffer<float, 2> {
  return make_points({{0.f, 0.f}, {10.f, 0.f}, {10.f, 10.f}, {0.f, 10.f}});
}

const std::array<std::array<make_cdt_index_t, 2>, 1> diagonal{
    std::array<make_cdt_index_t, 2>{0, 2}};
const std::array<std::array<make_cdt_index_t, 2>, 2> crossing_diagonals{
    std::array<make_cdt_index_t, 2>{0, 2},
    std::array<make_cdt_index_t, 2>{1, 3}};
const std::array<std::array<make_cdt_index_t, 2>, 5> outline_and_diagonal{
    std::array<make_cdt_index_t, 2>{0, 1},
    std::array<make_cdt_index_t, 2>{1, 2},
    std::array<make_cdt_index_t, 2>{2, 3},
    std::array<make_cdt_index_t, 2>{3, 0},
    std::array<make_cdt_index_t, 2>{0, 2}};
const std::array<bool, 5> outline_walls_only{true, true, true, true, false};

} // namespace

TEST_CASE("make_cdt without edges triangulates the hull", "[make_cdt]") {
  auto points = square();
  auto polys = tf::make_cdt(points.points());
  CHECK(polys.faces().size() == 2u);
  CHECK(polys.points().size() == 4u);
}

TEST_CASE("make_cdt recovers a constraint", "[make_cdt]") {
  auto points = square();
  // Every edge of the two-argument form is a region wall, so an interior
  // diagonal walls off half the square and only the odd-parity half stays.
  auto walled = tf::make_cdt(points.points(), tf::make_edges(diagonal));
  CHECK(walled.faces().size() == 1u);
  // A preserved-but-not-boundary diagonal inside a walled outline is held
  // as an edge without separating the interior.
  auto preserved =
      tf::make_cdt(points.points(), tf::make_edges(outline_and_diagonal),
                   tf::make_range(outline_walls_only));
  CHECK(preserved.faces().size() == 2u);
  CHECK(has_edge(preserved, 0, 2));
}

TEST_CASE("make_cdt binds a trailing bool as split_constraints",
          "[make_cdt]") {
  auto points = square();
  auto with_default = tf::make_cdt(points.points(), tf::make_edges(diagonal));
  auto with_bool =
      tf::make_cdt(points.points(), tf::make_edges(diagonal), true);
  CHECK(with_bool.faces().size() == with_default.faces().size());
  CHECK(with_bool.points().size() == with_default.points().size());
}

TEST_CASE("make_cdt resolves crossing constraints by default", "[make_cdt]") {
  auto points = square();
  // The crossing becomes a vertex; the four wall quadrants alternate
  // parity, so the two odd ones remain and both carry the created point.
  auto polys =
      tf::make_cdt(points.points(), tf::make_edges(crossing_diagonals));
  CHECK(polys.points().size() == 5u);
  CHECK(polys.faces().size() == 2u);
}

TEST_CASE("make_cdt refuses crossing constraints as an empty result",
          "[make_cdt]") {
  auto points = square();
  auto polys =
      tf::make_cdt(points.points(), tf::make_edges(crossing_diagonals), false);
  CHECK(polys.faces().size() == 0u);
  CHECK(polys.points().size() == 0u);
}

TEST_CASE("make_cdt takes the boundary mask as a range", "[make_cdt]") {
  auto points = square();
  auto with_default = tf::make_cdt(points.points(), tf::make_edges(diagonal));
  auto with_mask = tf::make_cdt(points.points(), tf::make_edges(diagonal),
                                tf::make_constant_range(true, 1));
  CHECK(with_mask.faces().size() == with_default.faces().size());

  auto refused =
      tf::make_cdt(points.points(), tf::make_edges(crossing_diagonals),
                   tf::make_constant_range(true, 2), false);
  CHECK(refused.faces().size() == 0u);
  CHECK(refused.points().size() == 0u);
}

TEST_CASE("make_cdt index-map overloads take the trailing bool",
          "[make_cdt]") {
  auto points = square();
  auto [polys, im] = tf::make_cdt(points.points(),
                                  tf::make_edges(crossing_diagonals),
                                  tf::return_index_map, false);
  CHECK(polys.faces().size() == 0u);

  auto [masked, masked_im] = tf::make_cdt(
      points.points(), tf::make_edges(outline_and_diagonal),
      tf::make_range(outline_walls_only), tf::return_index_map, true);
  CHECK(masked.faces().size() == 2u);
  CHECK(masked_im.f().size() == 4u);
}
