/**
 * @file test_census_wave.cpp
 * @brief The census: a green graph pays two passes, and the welds it
 *        states close into id classes.
 *
 * A clean pair is green on all three facts and states no weld. A box and
 * two copies of it moved by up to three float ulps are green pair by pair,
 * but the three near-duplicate carriers together create identities within
 * the quantum of one another and the first build pairs its rings oddly: the
 * census convicts, states its welds against the standing graph, and the diff
 * round they enter leaves the graph green. A box set through the side the
 * triple shares convicts the graph only where its crossings reach, and the
 * one round that states those sites alone leaves it green — four welds
 * where the convicted components' whole extent would state twenty-three,
 * so its bound fails the component grain. With four copies
 * the census runs several diff rounds, and a later round folds a group the
 * wave already took beside the world's group of the same key. The closure
 * joins every chain of pairs into one class, its members ascending, the
 * classes ordered by their smallest member.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg/expression/intersection.hpp>
#include <trueform/csg/expression/merge.hpp>
#include <trueform/csg/graph/census_conviction.hpp>
#include <trueform/csg/graph/make_census_classes.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/topology/triangulation_type.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

/// A hexahedron over eight corners, corner `v` at bits `x + 2y + 4z`, each
/// side split on its own diagonal and wound outward.
auto census_wave_hexahedron(const std::array<std::array<float, 3>, 8> &corners)
    -> tf::polygons_buffer<std::int32_t, float, 3, 3> {
  tf::polygons_buffer<std::int32_t, float, 3, 3> mesh;
  for (const auto &c : corners)
    mesh.points_buffer().emplace_back(c[0], c[1], c[2]);
  constexpr std::array<std::array<std::int32_t, 3>, 12> sides = {
      {{0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}, {0, 1, 5}, {0, 5, 4},
       {2, 6, 7}, {2, 7, 3}, {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}}};
  for (const auto &t : sides)
    mesh.faces_buffer().emplace_back(t[0], t[1], t[2]);
  return mesh;
}

} // namespace

TEMPLATE_TEST_CASE("census wave: a clean pair is green and states nothing",
                   "[csg][census]", (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  auto a = tf::make_sphere_mesh<index_t>(real_t(1), index_t(12), index_t(12));
  auto b = tf::make_sphere_mesh<index_t>(real_t(1), index_t(12), index_t(12));
  for (auto p : b.points())
    p[0] = p[0] + real_t(0.5);
  auto op0 = tf::test::make_tagged_operand(std::move(a));
  auto op1 = tf::test::make_tagged_operand(std::move(b));
  const auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                                    tf::test::no_sheets(), {});
  const auto conviction = tf::csg::graph::make_census_conviction(graph);
  REQUIRE(conviction.green());
  REQUIRE(conviction.failed == 0);
  REQUIRE(conviction.odd == 0);
  REQUIRE(conviction.self_sided == 0);
  REQUIRE(graph.census().stated == 0);
  REQUIRE(graph.census().rounds == 0);
  REQUIRE(graph.census().failed + graph.census().odd +
              graph.census().self_sided ==
          0);
  REQUIRE_FALSE(graph.census().capped);
}

TEST_CASE("census wave: a near-duplicate triple fires the census and the "
          "diff round is green",
          "[csg][census]") {
  using Catch::Matchers::WithinRel;
  auto op0 = tf::test::make_tagged_operand(
      census_wave_hexahedron({{{0.f, 0.f, 0.f},
                               {58.969505f, -79.6013f, 13.646595f},
                               {-11.219093f, -0.16019677f, 47.545376f},
                               {47.750412f, -79.761505f, 61.19197f},
                               {-20.432575f, -15.972463f, -4.8752103f},
                               {38.53693f, -95.57377f, 8.771385f},
                               {-31.651669f, -16.13266f, 42.670166f},
                               {27.317837f, -95.73396f, 56.31676f}}}));
  // the box with its corners moved by up to three float ulps, twice
  auto op1 = tf::test::make_tagged_operand(
      census_wave_hexahedron({{{0.f, 0.f, 0.f},
                               {58.9695f, -79.601295f, 13.646595f},
                               {-11.219092f, -0.16019675f, 47.545372f},
                               {47.75041f, -79.7615f, 61.19197f},
                               {-20.432575f, -15.972462f, -4.8752103f},
                               {38.53693f, -95.57377f, 8.771385f},
                               {-31.65167f, -16.132658f, 42.670166f},
                               {27.317835f, -95.73397f, 56.316757f}}}));
  auto op2 = tf::test::make_tagged_operand(
      census_wave_hexahedron({{{0.f, 0.f, 0.f},
                               {58.96951f, -79.6013f, 13.646595f},
                               {-11.219092f, -0.16019678f, 47.545372f},
                               {47.750416f, -79.761505f, 61.191967f},
                               {-20.432573f, -15.972463f, -4.875211f},
                               {38.536934f, -95.57376f, 8.771384f},
                               {-31.651667f, -16.132662f, 42.670166f},
                               {27.317837f, -95.733955f, 56.31676f}}}));
  const auto box_volume = double(tf::signed_volume(op0.mesh.polygons()));
  std::vector<tf::test::form_t<std::int32_t, float, 3>> forms{
      op0.form(), op1.form(), op2.form()};
  const auto graph = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::no_sheets(),
      tf::triangulation_type::cdt);
  REQUIRE(graph.census().stated > 0);
  REQUIRE(graph.census().rounds > 0);
  REQUIRE(tf::csg::graph::make_census_conviction(graph).green());
  for (const auto &expr :
       {tf::csg::merge(0, 1, 2), tf::csg::intersection(0, 1, 2)}) {
    const auto result = tf::test::csg_mesh_of(graph, expr);
    REQUIRE(result.polygons().size() > 0);
    REQUIRE(tf::is_closed(result.polygons()));
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE_THAT(double(tf::signed_volume(result.polygons())),
                 WithinRel(box_volume, 1e-5));
  }
}

TEST_CASE("census wave: a box on a near-duplicate triple heals at the sites "
          "its convictions name",
          "[csg][census]") {
  using Catch::Matchers::WithinRel;
  // the triple, and an axis-aligned box through the side the triple shares
  const std::array<std::array<std::array<float, 3>, 8>, 4> boxes = {
      {{{{0.f, 0.f, 0.f},
         {58.969505f, -79.6013f, 13.646595f},
         {-11.219093f, -0.16019677f, 47.545376f},
         {47.750412f, -79.761505f, 61.19197f},
         {-20.432575f, -15.972463f, -4.8752103f},
         {38.53693f, -95.57377f, 8.771385f},
         {-31.651669f, -16.13266f, 42.670166f},
         {27.317837f, -95.73396f, 56.31676f}}},
       {{{0.f, 0.f, 0.f},
         {58.9695f, -79.601295f, 13.646595f},
         {-11.219092f, -0.16019675f, 47.545372f},
         {47.75041f, -79.7615f, 61.19197f},
         {-20.432575f, -15.972462f, -4.8752103f},
         {38.53693f, -95.57377f, 8.771385f},
         {-31.65167f, -16.132658f, 42.670166f},
         {27.317835f, -95.73397f, 56.316757f}}},
       {{{0.f, 0.f, 0.f},
         {58.96951f, -79.6013f, 13.646595f},
         {-11.219092f, -0.16019678f, 47.545372f},
         {47.750416f, -79.761505f, 61.191967f},
         {-20.432573f, -15.972463f, -4.875211f},
         {38.536934f, -95.57376f, 8.771384f},
         {-31.651667f, -16.132662f, 42.670166f},
         {27.317837f, -95.733955f, 56.31676f}}},
       {{{-15.9702759f, -11.0817871f, 7.44110203f},
         {-3.97027588f, -11.0817871f, 7.44110203f},
         {-15.9702759f, 0.918212891f, 7.44110203f},
         {-3.97027588f, 0.918212891f, 7.44110203f},
         {-15.9702759f, -11.0817871f, 19.4411011f},
         {-3.97027588f, -11.0817871f, 19.4411011f},
         {-15.9702759f, 0.918212891f, 19.4411011f},
         {-3.97027588f, 0.918212891f, 19.4411011f}}}}};
  std::vector<tf::test::tagged_operand<std::int32_t, float>> operands;
  for (const auto &corners : boxes)
    operands.push_back(
        tf::test::make_tagged_operand(census_wave_hexahedron(corners)));
  const auto box_volume =
      double(tf::signed_volume(operands[0].mesh.polygons()));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::no_sheets(),
      tf::triangulation_type::cdt);
  const auto &census = graph.census();
  REQUIRE(census.rounds > 0);
  REQUIRE(census.stated > 0);
  // the blast radius: the sites the convictions name, never their components
  REQUIRE(census.stated <= 8);
  REQUIRE(census.failed + census.odd + census.self_sided == 0);
  REQUIRE_FALSE(census.capped);
  REQUIRE(graph.failed().size() == 0);
  const auto result = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
  REQUIRE(tf::is_closed(result.polygons()));
  REQUIRE(tf::is_manifold(result.polygons()));
  REQUIRE_THAT(double(tf::signed_volume(result.polygons())),
               WithinRel(box_volume, 1e-5));
}

TEST_CASE("census wave: a fold of a taking and the world's group keeps every "
          "carrier triangulated",
          "[csg][census]") {
  // the box with its corners moved by up to three float ulps, three times
  const std::array<std::array<std::array<float, 3>, 8>, 4> boxes = {
      {{{{0.f, 0.f, 0.f},
         {58.9695053f, -79.6013031f, 13.646595f},
         {-11.2190933f, -0.160196766f, 47.5453758f},
         {47.750412f, -79.7615051f, 61.1919708f},
         {-20.4325752f, -15.9724627f, -4.87521029f},
         {38.5369301f, -95.5737686f, 8.77138519f},
         {-31.6516685f, -16.1326599f, 42.670166f},
         {27.3178368f, -95.733963f, 56.316761f}}},
       {{{0.f, 0.f, 0.f},
         {58.9695129f, -79.6012802f, 13.6465979f},
         {-11.2190952f, -0.160196722f, 47.5453796f},
         {47.7504044f, -79.7615051f, 61.1919632f},
         {-20.4325752f, -15.9724646f, -4.87520885f},
         {38.5369377f, -95.5737915f, 8.77138424f},
         {-31.6516647f, -16.1326599f, 42.6701736f},
         {27.3178387f, -95.7339401f, 56.3167496f}}},
       {{{0.f, 0.f, 0.f},
         {58.9695015f, -79.601326f, 13.646594f},
         {-11.2190952f, -0.160196796f, 47.5453682f},
         {47.750412f, -79.7614975f, 61.1919746f},
         {-20.4325752f, -15.9724617f, -4.87520981f},
         {38.5369263f, -95.5737457f, 8.77138805f},
         {-31.6516685f, -16.1326656f, 42.6701622f},
         {27.317831f, -95.7339401f, 56.3167496f}}},
       {{{0.f, 0.f, 0.f},
         {58.9695015f, -79.601326f, 13.646595f},
         {-11.2190943f, -0.160196722f, 47.5453835f},
         {47.7504082f, -79.7614899f, 61.1919746f},
         {-20.4325695f, -15.9724655f, -4.87520885f},
         {38.5369377f, -95.5737534f, 8.77138424f},
         {-31.6516685f, -16.1326599f, 42.6701584f},
         {27.3178349f, -95.7339401f, 56.3167496f}}}}};
  std::vector<tf::test::tagged_operand<std::int32_t, float>> operands;
  for (const auto &corners : boxes)
    operands.push_back(
        tf::test::make_tagged_operand(census_wave_hexahedron(corners)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::no_sheets(),
      tf::triangulation_type::cdt);
  REQUIRE(graph.census().rounds > 0);
  REQUIRE(graph.failed().size() == 0);
  REQUIRE(graph.arrangement().arrangement().triangles().size() > 0);
}

TEST_CASE("census wave: pairs close into classes ordered by their smallest "
          "member",
          "[csg][census]") {
  tf::buffer<std::array<int, 3>> pairs;
  pairs.push_back({5, 9, 0});
  pairs.push_back({9, 2, 0});
  pairs.push_back({7, 7, 1}); // a singleton class is dropped
  pairs.push_back({40, 41, 1});
  const auto classes = tf::csg::graph::make_census_classes(pairs);
  REQUIRE(classes.size() == 2);
  REQUIRE(classes[0].size() == 3);
  REQUIRE(classes[0][0] == 2);
  REQUIRE(classes[0][1] == 5);
  REQUIRE(classes[0][2] == 9);
  REQUIRE(classes[1].size() == 2);
  REQUIRE(classes[1][0] == 40);
  REQUIRE(classes[1][1] == 41);
  // the weld count a round's fixed point is measured by
  REQUIRE(classes.data_buffer().size() - classes.size() == 3);
}
