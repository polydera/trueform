/**
 * @file test_tolerance_pulls.cpp
 * @brief What a band does to an arrangement: it only adds identifications.
 *
 * A vertex within the band of another form's plane is dropped exactly onto
 * the plane that face commits, and everything else is read as at zero. So an
 * exact fit keeps every record it states, a near-miss within the band welds
 * to the exact fit's topology, and a near-miss beyond the band births
 * nothing. The perpendicular family pins the boundary from both sides; what
 * the band did not move is exported as the input itself.
 *
 * An existing identification outranks a new one: an edge-edge contact the
 * input states survives whatever pulls the edges' ends — one end, both, the
 * touched edge's own form too, a chain of two contacts sharing an end — and
 * a pull that would undo it is forgone. Edges meeting at an end are a vertex
 * contact and hold nothing more: they leave the pull of the far end free.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>

#include "arrangement_builders.hpp"
#include "arrangement_readers.hpp"
#include "tagged_operand.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using pulls_index_t = int;
template <typename Real>
using pulls_mesh_t = tf::polygons_buffer<pulls_index_t, Real, 3, 3>;
using pulls_key_t = std::tuple<int, int, int, int, int, int, int, int>;

template <typename Real> auto pulls_box(double side) -> pulls_mesh_t<Real> {
  return tf::make_box_mesh<pulls_index_t>(Real(side), Real(side), Real(side));
}

template <typename Real>
auto pulls_quad(std::array<double, 3> a, std::array<double, 3> b,
                std::array<double, 3> c, std::array<double, 3> d)
    -> pulls_mesh_t<Real> {
  pulls_mesh_t<Real> mesh;
  for (const auto &q : {a, b, c, d})
    mesh.points_buffer().emplace_back(Real(q[0]), Real(q[1]), Real(q[2]));
  mesh.faces_buffer().emplace_back(0, 1, 2);
  mesh.faces_buffer().emplace_back(0, 2, 3);
  return mesh;
}

template <typename Real>
auto pulls_triangle(std::array<double, 3> a, std::array<double, 3> b,
                    std::array<double, 3> c) -> pulls_mesh_t<Real> {
  pulls_mesh_t<Real> mesh;
  for (const auto &q : {a, b, c})
    mesh.points_buffer().emplace_back(Real(q[0]), Real(q[1]), Real(q[2]));
  mesh.faces_buffer().emplace_back(0, 1, 2);
  return mesh;
}

/// A square sheet of side `width` in the plane `z = 0`, centred.
template <typename Real> auto pulls_sheet(double width) -> pulls_mesh_t<Real> {
  const double h = width / 2;
  return pulls_quad<Real>({-h, -h, 0}, {h, -h, 0}, {h, h, 0}, {-h, h, 0});
}

/// The cross-form records of an arrangement of `meshes` at `tolerance`.
template <typename Real>
auto pulls_records(std::vector<pulls_mesh_t<Real>> meshes, double tolerance)
    -> std::set<pulls_key_t> {
  std::vector<tf::test::tagged_operand<pulls_index_t, Real>> operands;
  operands.reserve(meshes.size());
  for (auto &mesh : meshes)
    operands.push_back(tf::test::make_tagged_operand(std::move(mesh)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms),
      tf::intersect_config{tf::intersect_mode::primitives, tolerance});
  std::set<pulls_key_t> keys;
  for (const auto &r : graph.world().intersections().flat_intersections())
    if (r.tag != r.tag_other)
      keys.insert({int(r.tag), int(r.object), int(r.target.label),
                   int(r.target.id), int(r.tag_other), int(r.object_other),
                   int(r.target_other.label), int(r.target_other.id)});
  return keys;
}

/// Records by the pair of feature kinds they join.
auto pulls_classes(const std::set<pulls_key_t> &keys)
    -> std::map<std::string, int> {
  std::map<std::string, int> out;
  for (const auto &k : keys)
    ++out[std::to_string(std::get<2>(k)) + "x" +
          std::to_string(std::get<6>(k))];
  return out;
}

constexpr double pulls_band = 1e-6;

/// The perpendicular family: a horizontal square and a vertical one whose
/// bottom edge stands `gap` above it.
template <typename Real>
auto pulls_perpendicular(double gap) -> std::vector<pulls_mesh_t<Real>> {
  return {pulls_quad<Real>({-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}),
          pulls_quad<Real>({0, -0.5, gap}, {0, 0.5, gap}, {0, 0.5, 1},
                           {0, -0.5, 1})};
}

} // namespace

TEMPLATE_TEST_CASE("tolerance pulls: an exact fit keeps every record under "
                   "the band",
                   "[arrangement][tolerance]", float, double) {
  // a sheet whose rim lies exactly on the walls of the box it spans
  const std::vector<pulls_mesh_t<TestType>> scene{pulls_box<TestType>(10),
                                                  pulls_sheet<TestType>(10)};
  const auto exact = pulls_records<TestType>(scene, 0.0);
  REQUIRE(exact.size() == 40);
  REQUIRE(pulls_records<TestType>(scene, pulls_band) == exact);
}

TEMPLATE_TEST_CASE("tolerance pulls: a sheet a third of the band short of the "
                   "walls welds to the exact fit",
                   "[arrangement][tolerance]", float, double) {
  const auto fit = pulls_classes(pulls_records<TestType>(
      {pulls_box<TestType>(10), pulls_sheet<TestType>(10)}, 0.0));
  const std::vector<pulls_mesh_t<TestType>> short_of{
      pulls_box<TestType>(10), pulls_sheet<TestType>(10 - 0.6 * pulls_band)};
  // the premise: at zero the sheet floats
  REQUIRE(pulls_records<TestType>(short_of, 0.0).empty());
  REQUIRE(pulls_classes(pulls_records<TestType>(short_of, pulls_band)) == fit);
}

TEMPLATE_TEST_CASE("tolerance pulls: a sheet a third of the band past the "
                   "walls welds to the exact fit",
                   "[arrangement][tolerance]", float, double) {
  const auto fit = pulls_classes(pulls_records<TestType>(
      {pulls_box<TestType>(10), pulls_sheet<TestType>(10)}, 0.0));
  const std::vector<pulls_mesh_t<TestType>> past{
      pulls_box<TestType>(10), pulls_sheet<TestType>(10 + 0.6 * pulls_band)};
  // the premise: at zero the sheet pierces the walls
  REQUIRE(pulls_classes(pulls_records<TestType>(past, 0.0)) != fit);
  REQUIRE(pulls_classes(pulls_records<TestType>(past, pulls_band)) == fit);
}

TEMPLATE_TEST_CASE("tolerance pulls: an edge a third of the band above a face "
                   "welds to the touch",
                   "[arrangement][tolerance]", float, double) {
  const auto touch = pulls_classes(
      pulls_records<TestType>(pulls_perpendicular<TestType>(0.0), 0.0));
  REQUIRE_FALSE(touch.empty());
  const auto close = pulls_perpendicular<TestType>(0.3 * pulls_band);
  REQUIRE(pulls_records<TestType>(close, 0.0).empty());
  REQUIRE(pulls_classes(pulls_records<TestType>(close, pulls_band)) == touch);
}

TEMPLATE_TEST_CASE("tolerance pulls: an edge one and a half bands above a face "
                   "births nothing",
                   "[arrangement][tolerance]", float, double) {
  const auto apart = pulls_perpendicular<TestType>(1.5 * pulls_band);
  REQUIRE(pulls_records<TestType>(apart, pulls_band).empty());
}

TEMPLATE_TEST_CASE("tolerance pulls: what the band did not move is exported "
                   "as the input",
                   "[arrangement][tolerance]", float, double) {
  // the sheet's corners are pulled onto the walls; the box's are not, and
  // its corners come out as the input's own coordinates, bit for bit
  auto box = pulls_box<TestType>(10);
  std::vector<std::array<TestType, 3>> corners;
  for (const auto &p : box.points())
    corners.push_back({p[0], p[1], p[2]});
  std::vector<tf::test::tagged_operand<pulls_index_t, TestType>> operands;
  operands.push_back(tf::test::make_tagged_operand(std::move(box)));
  operands.push_back(tf::test::make_tagged_operand(
      pulls_sheet<TestType>(10 - 0.6 * pulls_band)));
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_arrangement(
      tf::test::forms_range(forms),
      tf::intersect_config{tf::intersect_mode::primitives, pulls_band});
  REQUIRE(graph.lattice().moved_points().size() == 4);
  const auto mesh = tf::test::arrangement_mesh_of(graph);
  std::set<std::array<TestType, 3>> emitted;
  for (const auto &p : mesh.points())
    emitted.insert({p[0], p[1], p[2]});
  for (const auto &corner : corners)
    REQUIRE(emitted.count(corner) == 1);
}

/// The edge-edge scenes: A's edge (-1,0,1)-(1,0,1) touches B's top edge at
/// (0,0,1) exactly, and every other form stands a third of the band from
/// something it would pull.
template <typename Real>
auto pulls_edge_scene(const std::string &variant)
    -> std::vector<pulls_mesh_t<Real>> {
  const double z = 1 + 0.3 * pulls_band;
  std::vector<pulls_mesh_t<Real>> scene{
      pulls_quad<Real>({0, -1, -1}, {0, 1, -1}, {0, 1, 1}, {0, -1, 1}),
      pulls_triangle<Real>({-1, 0, 1}, {1, 0, 1}, {0, 0.5, 1.5})};
  if (variant == "one end")
    scene.push_back(
        pulls_quad<Real>({-2, -1, z}, {-0.5, -1, z}, {-0.5, 1, z}, {-2, 1, z}));
  else if (variant == "both ends") {
    scene.push_back(
        pulls_quad<Real>({-2, -1, z}, {-0.5, -1, z}, {-0.5, 1, z}, {-2, 1, z}));
    scene.push_back(
        pulls_quad<Real>({0.5, -1, z}, {2, -1, z}, {2, 1, z}, {0.5, 1, z}));
  } else if (variant == "both edges")
    scene.push_back(
        pulls_quad<Real>({-2, -2, z}, {2, -2, z}, {2, 2, z}, {-2, 2, z}));
  else {
    // A's second edge touches D's top edge at (0.5, 0.25, 1.25), and the
    // end the two touching edges share is the one pulled
    scene.push_back(pulls_quad<Real>({0.5, -1, -1}, {0.5, 1, -1},
                                     {0.5, 1, 1.25}, {0.5, -1, 1.25}));
    scene.push_back(
        pulls_quad<Real>({0.8, -1, z}, {2, -1, z}, {2, 1, z}, {0.8, 1, z}));
  }
  return scene;
}

TEMPLATE_TEST_CASE("tolerance pulls: an edge-edge contact survives every pull "
                   "on its edges",
                   "[arrangement][tolerance]", float, double) {
  for (const std::string variant :
       {"one end", "both ends", "both edges", "a chain"}) {
    CAPTURE(variant);
    const auto scene = pulls_edge_scene<TestType>(variant);
    const auto exact = pulls_records<TestType>(scene, 0.0);
    const auto banded = pulls_records<TestType>(scene, pulls_band);
    int edge_edge = 0;
    for (const auto &key : exact) {
      REQUIRE(banded.count(key) == 1);
      edge_edge += std::get<2>(key) == 2 && std::get<6>(key) == 2 ? 1 : 0;
    }
    // the premise: the scene states the edge-edge contacts at zero
    REQUIRE(edge_edge >= (variant == "a chain" ? 4 : 2));
  }
}

/// A fan whose apex stands `lift` above a sheet and whose rim stands exactly
/// on the corners of a third form's triangle, so every fan edge meets a
/// triangle edge at a shared end.
template <typename Real>
auto pulls_fan_scene(double lift) -> std::vector<pulls_mesh_t<Real>> {
  pulls_mesh_t<Real> fan;
  for (const auto &q : std::vector<std::array<double, 3>>{
           {0.25, -0.375, lift}, {-1, -1, 1}, {1, -1, 1}, {0, 1, 1}})
    fan.points_buffer().emplace_back(Real(q[0]), Real(q[1]), Real(q[2]));
  fan.faces_buffer().emplace_back(0, 1, 2);
  fan.faces_buffer().emplace_back(0, 2, 3);
  fan.faces_buffer().emplace_back(0, 3, 1);
  return {pulls_quad<Real>({-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}),
          std::move(fan),
          pulls_triangle<Real>({-1, -1, 1}, {1, -1, 1}, {0, 1, 1})};
}

TEMPLATE_TEST_CASE("tolerance pulls: edges meeting at a shared end leave the "
                   "far end's pull free",
                   "[arrangement][tolerance]", float, double) {
  const auto touch =
      pulls_classes(pulls_records<TestType>(pulls_fan_scene<TestType>(0), 0.0));
  const auto close = pulls_fan_scene<TestType>(0.3 * pulls_band);
  // the premise: at zero the apex floats
  REQUIRE(pulls_classes(pulls_records<TestType>(close, 0.0)) != touch);
  REQUIRE(pulls_classes(pulls_records<TestType>(close, pulls_band)) == touch);
}
