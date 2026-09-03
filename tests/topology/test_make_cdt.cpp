#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/views/constant.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/reindex/by_mask.hpp>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/topology/cdt_config.hpp>
#include <trueform/topology/cdt_region_mode.hpp>
#include <trueform/topology/make_cdt.hpp>
#include <trueform/topology/return_region_labels.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <set>
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

// A walled square holding a walled square hole, with unconstrained hull
// corners beyond both, so the hull-exterior band owns triangles: points
// 0-3 the outer wall, 4-7 the hole wall, 8-11 the hull.
auto holed_square_points() -> tf::points_buffer<float, 2> {
  return make_points({{2.f, 2.f},
                      {8.f, 2.f},
                      {8.f, 8.f},
                      {2.f, 8.f},
                      {4.f, 4.f},
                      {6.f, 4.f},
                      {6.f, 6.f},
                      {4.f, 6.f},
                      {0.f, 0.f},
                      {10.f, 0.f},
                      {10.f, 10.f},
                      {0.f, 10.f}});
}

const std::array<std::array<make_cdt_index_t, 2>, 8> holed_square_edges{
    std::array<make_cdt_index_t, 2>{0, 1},
    std::array<make_cdt_index_t, 2>{1, 2},
    std::array<make_cdt_index_t, 2>{2, 3},
    std::array<make_cdt_index_t, 2>{3, 0},
    std::array<make_cdt_index_t, 2>{4, 5},
    std::array<make_cdt_index_t, 2>{5, 6},
    std::array<make_cdt_index_t, 2>{6, 7},
    std::array<make_cdt_index_t, 2>{7, 4}};

// A ring inside a hole inside a ring: points 0-3 the outer ring's outer
// wall (also the hull), 4-7 its inner wall, 8-11 the island ring's outer
// wall, 12-15 its inner wall.
auto ringed_island_points() -> tf::points_buffer<float, 2> {
  return make_points({{0.f, 0.f},
                      {20.f, 0.f},
                      {20.f, 20.f},
                      {0.f, 20.f},
                      {4.f, 4.f},
                      {16.f, 4.f},
                      {16.f, 16.f},
                      {4.f, 16.f},
                      {8.f, 8.f},
                      {12.f, 8.f},
                      {12.f, 12.f},
                      {8.f, 12.f},
                      {9.f, 9.f},
                      {11.f, 9.f},
                      {11.f, 11.f},
                      {9.f, 11.f}});
}

const std::array<std::array<make_cdt_index_t, 2>, 16> ringed_island_edges{
    std::array<make_cdt_index_t, 2>{0, 1},
    std::array<make_cdt_index_t, 2>{1, 2},
    std::array<make_cdt_index_t, 2>{2, 3},
    std::array<make_cdt_index_t, 2>{3, 0},
    std::array<make_cdt_index_t, 2>{4, 5},
    std::array<make_cdt_index_t, 2>{5, 6},
    std::array<make_cdt_index_t, 2>{6, 7},
    std::array<make_cdt_index_t, 2>{7, 4},
    std::array<make_cdt_index_t, 2>{8, 9},
    std::array<make_cdt_index_t, 2>{9, 10},
    std::array<make_cdt_index_t, 2>{10, 11},
    std::array<make_cdt_index_t, 2>{11, 8},
    std::array<make_cdt_index_t, 2>{12, 13},
    std::array<make_cdt_index_t, 2>{13, 14},
    std::array<make_cdt_index_t, 2>{14, 15},
    std::array<make_cdt_index_t, 2>{15, 12}};

// The output points come back in the triangulator's welded order, not the
// input order, so a triangle is classified by where it lies: its centroid
// is strictly interior, and every region here is a nest of axis-aligned
// squares no triangle may cross.
template <typename Polygons>
auto face_centroid(const Polygons &polys, std::size_t i)
    -> std::array<float, 2> {
  auto face = polys.faces()[i];
  std::array<float, 2> c{0.f, 0.f};
  for (int k = 0; k < 3; ++k) {
    c[0] += polys.points()[std::size_t(face[std::size_t(k)])][0];
    c[1] += polys.points()[std::size_t(face[std::size_t(k)])][1];
  }
  c[0] /= 3.f;
  c[1] /= 3.f;
  return c;
}

auto centroid_inside(const std::array<float, 2> &c, float lo, float hi)
    -> bool {
  return c[0] > lo && c[0] < hi && c[1] > lo && c[1] < hi;
}

// 0 = hull band, 1 = annulus, 2 = hole.
template <typename Polygons>
auto holed_square_region(const Polygons &polys, std::size_t i) -> int {
  auto c = face_centroid(polys, i);
  if (centroid_inside(c, 4.f, 6.f))
    return 2;
  if (centroid_inside(c, 2.f, 8.f))
    return 1;
  return 0;
}

// 0 = outer ring body, 1 = hole, 2 = island ring body, 3 = island's hole.
template <typename Polygons>
auto ringed_island_region(const Polygons &polys, std::size_t i) -> int {
  auto c = face_centroid(polys, i);
  if (centroid_inside(c, 9.f, 11.f))
    return 3;
  if (centroid_inside(c, 8.f, 12.f))
    return 2;
  if (centroid_inside(c, 4.f, 16.f))
    return 1;
  return 0;
}

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

TEST_CASE("make_cdt binds a trailing bool as split_constraints", "[make_cdt]") {
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

TEST_CASE("make_cdt index-map overloads take the trailing bool", "[make_cdt]") {
  auto points = square();
  auto [polys, im] =
      tf::make_cdt(points.points(), tf::make_edges(crossing_diagonals),
                   tf::return_index_map, false);
  CHECK(polys.faces().size() == 0u);

  auto [masked, masked_im] = tf::make_cdt(
      points.points(), tf::make_edges(outline_and_diagonal),
      tf::make_range(outline_walls_only), tf::return_index_map, true);
  CHECK(masked.faces().size() == 2u);
  CHECK(masked_im.f().size() == 4u);
}

TEST_CASE("make_cdt region labels state the nesting parity of a holed square",
          "[make_cdt]") {
  auto points = holed_square_points();
  auto [polys, labels] =
      tf::make_cdt(points.points(), tf::make_edges(holed_square_edges),
                   tf::return_region_labels);

  REQUIRE(polys.faces().size() > 0u);
  REQUIRE(labels.size() == polys.faces().size());
  CHECK(polys.points().size() == 12u);

  std::set<make_cdt_index_t> distinct(labels.begin(), labels.end());
  CHECK(distinct == std::set<make_cdt_index_t>{0, 1});

  bool saw_hole = false;
  for (std::size_t i = 0; i < polys.faces().size(); ++i) {
    switch (holed_square_region(polys, i)) {
    case 2:
      saw_hole = true;
      CHECK(labels[i] == 0);
      break;
    case 1:
      CHECK(labels[i] == 1);
      break;
    default:
      CHECK(labels[i] == 0);
    }
  }
  CHECK(saw_hole);
}

TEST_CASE("make_cdt component labels tell the hole from the exterior",
          "[make_cdt]") {
  auto points = holed_square_points();
  auto [polys, labels] =
      tf::make_cdt(points.points(), tf::make_edges(holed_square_edges),
                   tf::return_region_labels, tf::cdt_region_mode::components);

  REQUIRE(labels.size() == polys.faces().size());
  std::set<make_cdt_index_t> distinct(labels.begin(), labels.end());
  CHECK(distinct == std::set<make_cdt_index_t>{0, 1, 2});

  make_cdt_index_t hole_label = -1;
  make_cdt_index_t annulus_label = -1;
  for (std::size_t i = 0; i < polys.faces().size(); ++i) {
    switch (holed_square_region(polys, i)) {
    case 2:
      if (hole_label == -1)
        hole_label = labels[i];
      CHECK(labels[i] == hole_label);
      break;
    case 1:
      if (annulus_label == -1)
        annulus_label = labels[i];
      CHECK(labels[i] == annulus_label);
      break;
    default:
      CHECK(labels[i] == 0);
    }
  }
  CHECK(hole_label > 0);
  CHECK(annulus_label > 0);
  CHECK(hole_label != annulus_label);
}

TEST_CASE("make_cdt labels separate the depth-2 island the interior read "
          "cannot express",
          "[make_cdt]") {
  auto points = ringed_island_points();

  // Nesting is a parity: the island ring body reads 1 like the outer ring
  // body, and both holes read 0 like the exterior — the island's identity
  // is what the default parity filter silently flattens.
  auto [nested, nesting_labels] =
      tf::make_cdt(points.points(), tf::make_edges(ringed_island_edges),
                   tf::return_region_labels);
  REQUIRE(nesting_labels.size() == nested.faces().size());
  const std::array<make_cdt_index_t, 4> parity_of_region{1, 0, 1, 0};
  for (std::size_t i = 0; i < nested.faces().size(); ++i)
    CHECK(nesting_labels[i] ==
          parity_of_region[std::size_t(ringed_island_region(nested, i))]);

  // Components state the ids parity cannot: each of the four regions is
  // its own wall-cut component, the island distinct from every other.
  auto [cut, component_labels] =
      tf::make_cdt(points.points(), tf::make_edges(ringed_island_edges),
                   tf::return_region_labels, tf::cdt_region_mode::components);
  REQUIRE(component_labels.size() == cut.faces().size());
  std::array<make_cdt_index_t, 4> label_of_region{-1, -1, -1, -1};
  for (std::size_t i = 0; i < cut.faces().size(); ++i) {
    auto region = std::size_t(ringed_island_region(cut, i));
    if (label_of_region[region] == -1)
      label_of_region[region] = component_labels[i];
    CHECK(component_labels[i] == label_of_region[region]);
  }
  std::set<make_cdt_index_t> distinct(label_of_region.begin(),
                                      label_of_region.end());
  CHECK(distinct.size() == 4u);
  // The hull outline is walled, so the hull-exterior component 0 owns no
  // triangle and every region carries its own positive id.
  for (auto label : label_of_region)
    CHECK(label > 0);
}

TEST_CASE("make_cdt labels come back empty when preserve mode refuses",
          "[make_cdt]") {
  auto points = square();
  auto [polys, labels] =
      tf::make_cdt(points.points(), tf::make_edges(crossing_diagonals),
                   tf::return_region_labels, false);
  CHECK(polys.faces().size() == 0u);
  CHECK(polys.points().size() == 0u);
  CHECK(labels.size() == 0u);
}

TEST_CASE("make_cdt default entry equals the labels read filtered by parity",
          "[make_cdt]") {
  auto points = holed_square_points();
  auto edges = tf::make_edges(holed_square_edges);

  auto interior_only = tf::make_cdt(points.points(), edges);
  auto [full, labels] =
      tf::make_cdt(points.points(), edges, tf::return_region_labels);

  auto interior = tf::make_mapped_range(
      tf::make_range(labels), [](auto label) { return label % 2 == 1; });
  auto filtered = tf::reindexed_by_mask<make_cdt_index_t>(
      tf::make_polygons(full.faces(), full.points()), interior);

  REQUIRE(interior_only.faces().size() == filtered.faces().size());
  for (std::size_t i = 0; i < filtered.faces().size(); ++i)
    for (int c = 0; c < 3; ++c)
      CHECK(interior_only.faces()[i][std::size_t(c)] ==
            filtered.faces()[i][std::size_t(c)]);
  REQUIRE(interior_only.points().size() == filtered.points().size());
  for (std::size_t i = 0; i < filtered.points().size(); ++i)
    for (int c = 0; c < 2; ++c)
      CHECK(interior_only.points()[i][std::size_t(c)] ==
            filtered.points()[i][std::size_t(c)]);
}

TEST_CASE("make_cdt region labels compose with the mask and the index map",
          "[make_cdt]") {
  auto points = square();
  auto [polys, labels, im] =
      tf::make_cdt(points.points(), tf::make_edges(outline_and_diagonal),
                   tf::make_range(outline_walls_only), tf::return_region_labels,
                   tf::return_index_map);
  // The preserved diagonal never separates regions, so both triangles of
  // the walled square read nesting parity 1 — and nothing is filtered.
  REQUIRE(polys.faces().size() == 2u);
  REQUIRE(labels.size() == 2u);
  CHECK(labels[0] == 1);
  CHECK(labels[1] == 1);
  CHECK(im.f().size() == 4u);
  CHECK(im.kept_ids().size() == polys.points().size());
}
