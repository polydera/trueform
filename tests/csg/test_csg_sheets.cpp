/**
 * @file test_csg_sheets.cpp
 * @brief Sheet semantics: forms declared as sheets act as oriented
 *        separators in the boolean algebra.
 *
 * A sheet's fragments always divide their two sides (no Mode-2
 * self-merge) and its operand bit means "on the back side of the
 * sheet's normal", so plain difference/intersection cut volumes against
 * it into closed, capped halves. Volume forms keep today's semantics;
 * declaring a clean closed mesh a sheet is a no-op.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/csg/graph/winding_side.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using sheets_index_t = int;
using sheets_real_t = double;
using sheets_mesh_t = tf::polygons_buffer<sheets_index_t, sheets_real_t, 3, 3>;

constexpr double sheets_pi = tf::pi<double>;

auto sheets_frame_at(sheets_real_t x, sheets_real_t y, sheets_real_t z) {
  return tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<sheets_real_t, 3>{x, y, z}));
}

using sheets_frame_t = decltype(sheets_frame_at(0, 0, 0));
using sheets_operand_t =
    tf::test::tagged_operand<sheets_index_t, sheets_real_t>;
using sheets_form_t = tf::test::form_t<sheets_index_t, sheets_real_t, 3>;

struct graph_holder {
  std::vector<sheets_operand_t> operands;
  std::vector<sheets_form_t> forms;
  std::vector<int> sheets;
  decltype(tf::test::build_range_csg_graph(
      tf::test::forms_range(std::declval<std::vector<sheets_form_t> &>()),
      tf::test::no_sheets(), tf::arrangement_config{})) graph;

  graph_holder(std::vector<sheets_operand_t> ops, std::vector<int> s)
      : operands(std::move(ops)), forms(tf::test::tagged_forms(operands)),
        sheets(std::move(s)),
        graph(tf::test::build_range_csg_graph(
            tf::test::forms_range(forms), tf::test::sheets_of(sheets), {})) {}

  graph_holder(const graph_holder &) = delete;
  graph_holder &operator=(const graph_holder &) = delete;
};

auto sheets_volume_of(const sheets_mesh_t &m) -> double {
  return m.size() ? double(tf::signed_volume(m.polygons())) : 0.0;
}

// z=0 wall tag of a cell: -2 = no wall faces, -1 = mixed tags, else
// the single tag every wall face carries
template <typename TagOfFace>
auto wall_tag_of(const sheets_mesh_t &cell, const TagOfFace &tag_of_face)
    -> sheets_index_t {
  auto pts = cell.polygons().points();
  auto faces = cell.faces();
  sheets_index_t lone = -2;
  for (std::size_t f = 0; f < faces.size(); ++f) {
    auto face = faces[f];
    bool wall = true;
    for (int j = 0; j < 3; ++j)
      wall = wall && std::abs(double(pts[face[j]][2])) < 1e-12;
    if (!wall)
      continue;
    const sheets_index_t t = tag_of_face(f);
    lone = lone == -2 || lone == t ? t : sheets_index_t(-1);
  }
  return lone;
}

auto centroid_z(const sheets_mesh_t &m) -> double {
  double z = 0;
  std::size_t n = 0;
  for (auto p : m.points()) {
    z += double(p[2]);
    ++n;
  }
  return n ? z / double(n) : 0.0;
}

void check_closed_solid(const sheets_mesh_t &m, double expected_volume,
                        double tol) {
  REQUIRE(m.size() > 0);
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
  REQUIRE_THAT(sheets_volume_of(m),
               Catch::Matchers::WithinAbs(expected_volume, tol));
}

auto centroid_y(const sheets_mesh_t &m) -> double {
  double y = 0;
  std::size_t n = 0;
  for (auto p : m.points()) {
    y += double(p[1]);
    ++n;
  }
  return n ? y / double(n) : 0.0;
}

auto n_boundary(const sheets_mesh_t &m) -> std::size_t {
  return tf::make_boundary_edges(m.polygons()).size();
}

// (y, z) -> (-z, y): a +z normal becomes -y
auto rotated_about_x(sheets_mesh_t m) -> sheets_mesh_t {
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i) {
    auto p = m.points_buffer()[i];
    const sheets_real_t y = p[1], z = p[2];
    m.points_buffer()[i][1] = -z;
    m.points_buffer()[i][2] = y;
  }
  return m;
}

auto shifted_x(sheets_mesh_t m, sheets_real_t dx) -> sheets_mesh_t {
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i)
    m.points_buffer()[i][0] += dx;
  return m;
}

// every face's normal has the given sign along `axis`
auto faces_point(const sheets_mesh_t &m, int axis, double sign) -> bool {
  auto pts = m.polygons().points();
  for (auto face : m.faces()) {
    const auto n =
        tf::cross(pts[face[1]] - pts[face[0]], pts[face[2]] - pts[face[0]]);
    if (double(n[axis]) * sign <= 0.0)
      return false;
  }
  return m.size() > 0;
}

} // namespace

// ============================================================================
// A sheet plane cuts a box into closed capped halves.
// ============================================================================
TEST_CASE("sheets: plane cuts a box into closed halves", "[csg][sheets]") {
  auto box = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2)); // vol 8
  auto plane = tf::make_plane_mesh<sheets_index_t>(sheets_real_t(4),
                                                   sheets_real_t(4)); // z=0, +z

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1});
  auto &graph = holder.graph;

  SECTION("difference keeps the half above the normal") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
    check_closed_solid(m, 4.0, 1e-9);
    REQUIRE(centroid_z(m) > 0.0);
  }
  SECTION("intersection keeps the half below the normal") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
    check_closed_solid(m, 4.0, 1e-9);
    REQUIRE(centroid_z(m) < 0.0);
  }
  SECTION("union and sheet-minus-volume are honest open boundaries") {
    auto u = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
    REQUIRE(u.size() > 0);
    REQUIRE_FALSE(tf::is_closed(u.polygons()));
    auto r = tf::test::csg_mesh_of(graph, tf::csg::difference(1, 0));
    REQUIRE(r.size() > 0);
    REQUIRE_FALSE(tf::is_closed(r.polygons()));
  }
}

// ============================================================================
// A sheet plane cuts a sphere: the original ask.
// ============================================================================
TEST_CASE("sheets: plane cuts a sphere into closed halves", "[csg][sheets]") {
  auto sphere = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(1), 96, 96);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(3), sheets_real_t(3));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      sphere, tf::transformation<sheets_real_t, 3>(
                  sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1});
  auto &graph = holder.graph;

  const double half = (4.0 / 3.0) * sheets_pi / 2.0;
  auto upper = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
  auto lower = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
  check_closed_solid(upper, half, half * 0.01);
  check_closed_solid(lower, half, half * 0.01);
  REQUIRE(centroid_z(upper) > 0.0);
  REQUIRE(centroid_z(lower) < 0.0);
}

// ============================================================================
// A clean closed mesh declared as a sheet is a no-op.
// ============================================================================
TEST_CASE("sheets: closed mesh as sheet is a no-op", "[csg][sheets]") {
  auto a = tf::make_box_mesh<sheets_index_t>(sheets_real_t(2), sheets_real_t(2),
                                             sheets_real_t(2));
  auto b = tf::make_box_mesh<sheets_index_t>(sheets_real_t(2), sheets_real_t(2),
                                             sheets_real_t(2));

  auto make_forms = [&] {
    std::vector<sheets_operand_t> operands;
    operands.push_back(tf::test::make_tagged_operand(
        a, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        b, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(1, 0, 0).transformation())));
    return operands;
  };

  auto volumes = std::vector<std::pair<tf::csg::expr, double>>{
      {tf::csg::merge(0, 1), 12.0},
      {tf::csg::intersection(0, 1), 4.0},
      {tf::csg::difference(0, 1), 4.0},
      {tf::csg::difference(1, 0), 4.0},
  };

  graph_holder as_sheet(make_forms(), {1});
  graph_holder plain_holder(make_forms(), {});
  auto &plain = plain_holder.graph;

  for (auto &[e, expected] : volumes) {
    auto m_sheet = tf::test::csg_mesh_of(as_sheet.graph, e);
    auto m_plain = tf::test::csg_mesh_of(plain, e);
    check_closed_solid(m_sheet, expected, 1e-9);
    REQUIRE(m_sheet.size() == m_plain.size());
    REQUIRE_THAT(sheets_volume_of(m_sheet),
                 Catch::Matchers::WithinAbs(sheets_volume_of(m_plain), 1e-12));
  }
}

// ============================================================================
// A floater behind the sheet that nothing encloses joins the outside: its
// envelope must not survive enumeration as an inverted shell.
// ============================================================================
TEST_CASE("sheets: un-enclosed behind-side floater leaves no pseudo-domain",
          "[csg][sheets]") {
  auto straddle =
      tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(1), 32, 32);
  auto below = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(0.5), 32, 32);
  tf::ensure_positive_orientation(straddle.polygons());
  tf::ensure_positive_orientation(below.polygons());
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(4), sheets_real_t(4));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      straddle, tf::transformation<sheets_real_t, 3>(
                    sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      below, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, -2).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {2});

  auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
  REQUIRE(cells.size() == 3); // two halves + the floater
  for (auto &cell : cells) {
    REQUIRE(cell.polygons().size() > 0);
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(double(tf::signed_volume(cell.polygons())) > 0.0);
  }
}

// ============================================================================
// Disconnected volumes classify by side of the sheet.
// ============================================================================
TEST_CASE("sheets: floating volumes know their side", "[csg][sheets]") {
  auto above = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(1), sheets_real_t(1), sheets_real_t(1)); // vol 1
  auto below = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(1), sheets_real_t(1), sheets_real_t(1));
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      above, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 2).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      below, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, -2).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {2});
  auto &graph = holder.graph;

  SECTION("above box is entirely above") {
    check_closed_solid(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 2)),
                       1.0, 1e-9);
    REQUIRE(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 2)).size() ==
            0);
  }
  SECTION("below box is entirely below") {
    check_closed_solid(
        tf::test::csg_mesh_of(graph, tf::csg::intersection(1, 2)), 1.0, 1e-9);
    REQUIRE(tf::test::csg_mesh_of(graph, tf::csg::difference(1, 2)).size() ==
            0);
  }
}

// ============================================================================
// A floater far below, near the sheet's footprint corner: the sheet
// subtends well under quarter winding there — the side must still be
// the winding sign. (Regression: a w > 1/4 threshold classified this
// as above.)
// ============================================================================
TEST_CASE("sheets: far corner floater still classifies by sign",
          "[csg][sheets]") {
  auto box = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(1), sheets_real_t(1), sheets_real_t(1));
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(2.6, -2.6, -2.5).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1});
  auto &graph = holder.graph;

  check_closed_solid(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)),
                     1.0, 1e-9);
  REQUIRE(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)).size() == 0);
}

// ============================================================================
// Sheets compose with volume expressions.
// ============================================================================
TEST_CASE("sheets: cut of an intersection", "[csg][sheets]") {
  auto a = tf::make_box_mesh<sheets_index_t>(sheets_real_t(2), sheets_real_t(2),
                                             sheets_real_t(2));
  auto b = tf::make_box_mesh<sheets_index_t>(sheets_real_t(2), sheets_real_t(2),
                                             sheets_real_t(2));
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      a, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      b, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(1, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {2});
  auto &graph = holder.graph;

  // (a ^ b) has volume 4; the sheet at z=0 halves it.
  auto upper = tf::test::csg_mesh_of(
      graph, tf::csg::difference(tf::csg::intersection(0, 1), 2));
  check_closed_solid(upper, 2.0, 1e-9);
  REQUIRE(centroid_z(upper) > 0.0);
}

// ============================================================================
// Without a sheet declaration an open plane is inert: volume semantics
// fuse open fragments away, so it neither cuts nor contains.
// ============================================================================
TEST_CASE("sheets: undeclared open plane is inert", "[csg][sheets]") {
  auto sphere = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(1), 48, 48);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));
  const double full = double(tf::signed_volume(sphere.polygons()));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      sphere, tf::transformation<sheets_real_t, 3>(
                  sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  check_closed_solid(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
                     full, 1e-9);
  REQUIRE(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)).size() ==
          0);
}

namespace {

auto area_of(const sheets_mesh_t &m) -> double {
  double a = 0;
  auto polys = m.polygons();
  for (std::size_t f = 0; f < polys.size(); ++f) {
    auto poly = polys[f];
    tf::vector<sheets_real_t, 3> e0 = poly[1] - poly[0];
    tf::vector<sheets_real_t, 3> e1 = poly[2] - poly[0];
    a += double(tf::cross(e0, e1).length()) / 2;
  }
  return a;
}

} // namespace

// ============================================================================
// op(sheet) is the half-space behind the sheet's normal, used
// volumetrically: sheet-led expressions select regions, and the mesh is
// that region's boundary — closed when bounded, honestly open when not.
// ============================================================================
TEST_CASE("sheets: sheet-led expressions are volumetric half-spaces",
          "[csg][sheets]") {
  auto sphere = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(1), 96, 96);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));
  const double full = double(tf::signed_volume(sphere.polygons()));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      sphere, tf::transformation<sheets_real_t, 3>(
                  sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1});
  auto &graph = holder.graph;

  SECTION("plane & sphere = lower half-ball: hemisphere + cap disc") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::intersection(1, 0));
    check_closed_solid(m, full / 2, 0.01 * full);
    REQUIRE(centroid_z(m) < 0.0);
    REQUIRE_THAT(area_of(m), Catch::Matchers::WithinAbs(3 * sheets_pi,
                                                        0.01 * 3 * sheets_pi));
  }
  SECTION("plane - sphere = boundary of the unbounded region below") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::difference(1, 0));
    REQUIRE(m.size() > 0);
    REQUIRE_FALSE(tf::is_closed(m.polygons()));
    // punctured plane (36 - pi) + hemisphere wall (2 pi)
    REQUIRE_THAT(area_of(m), Catch::Matchers::WithinAbs(
                                 36 + sheets_pi, 0.01 * (36 + sheets_pi)));
  }
}

// ============================================================================
// A composite expression over disconnected volumes: the sheet halves the
// straddling sphere while whole floaters keep or lose membership by
// side, each output piece closed on its own.
// ============================================================================
TEST_CASE("sheets: composite cut of disconnected volumes", "[csg][sheets]") {
  auto straddle =
      tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(1), 48, 48);
  auto above = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(0.5), 48, 48);
  auto below = tf::make_sphere_mesh<sheets_index_t>(sheets_real_t(0.5), 48, 48);
  for (auto *s : {&straddle, &above, &below})
    tf::ensure_positive_orientation(s->polygons());
  auto plane =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(6), sheets_real_t(6));
  const double full = double(tf::signed_volume(straddle.polygons()));
  const double small = double(tf::signed_volume(above.polygons()));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      straddle, tf::transformation<sheets_real_t, 3>(
                    sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      above, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 2).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      below, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, -2).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {3});
  auto &graph = holder.graph;

  auto solids = tf::csg::merge(tf::csg::merge(0, 1), 2);
  auto check_two_pieces = [&](const sheets_mesh_t &m, double sign) {
    REQUIRE_THAT(sheets_volume_of(m),
                 Catch::Matchers::WithinAbs(full / 2 + small, 0.01 * full));
    auto [labels, n] = tf::make_manifold_edge_connected_component_labels(
        m.polygons());
    REQUIRE(n == sheets_index_t(2));
    auto [pieces, ids] = tf::split_into_components(m.polygons(), labels);
    for (auto &piece : pieces) {
      REQUIRE(tf::is_closed(piece.polygons()));
      REQUIRE(tf::is_manifold(piece.polygons()));
      REQUIRE(sign * centroid_z(piece) > 0.0);
      const double v = sheets_volume_of(piece);
      const bool is_half = std::abs(v - full / 2) < 0.01 * full;
      const bool is_floater = std::abs(v - small) < 0.01 * small;
      REQUIRE((is_half || is_floater));
    }
  };

  SECTION("difference keeps upper half + above floater") {
    check_two_pieces(
        tf::test::csg_mesh_of(graph, tf::csg::difference(solids, 3)), +1.0);
  }
  SECTION("intersection keeps lower half + below floater") {
    check_two_pieces(
        tf::test::csg_mesh_of(graph, tf::csg::intersection(solids, 3)), -1.0);
  }
}

// ============================================================================
// Coincident duplicate sheets: the outside must stay excluded.
// ============================================================================

namespace {

auto reversed_copy(const sheets_mesh_t &m) -> sheets_mesh_t {
  sheets_mesh_t r = m;
  for (auto &&face : r.faces_buffer())
    std::swap(face[0], face[2]);
  return r;
}

} // namespace

TEST_CASE("sheets: coincident duplicate sheet keeps the outside excluded",
          "[csg][sheets]") {
  // Two coincident copies of the cutting plane. With opposite windings
  // they tile space ("everything is behind one of them"), so the
  // universe cannot be detected as the all-false row unless fusing a
  // sheet's dangling halves also clears that sheet's bit. Either
  // winding must give exactly the two box halves.
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2)); // vol 8
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(4), sheets_real_t(4)); // z=0, +z
  sheets_mesh_t plane = tf::triangulated(planeq.polygons());
  sheets_mesh_t plane_rev = reversed_copy(plane);

  auto check_halves = [&](sheets_mesh_t &second_sheet) {
    std::vector<sheets_operand_t> operands;
    operands.push_back(tf::test::make_tagged_operand(
        box, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        second_sheet, tf::transformation<sheets_real_t, 3>(
                          sheets_frame_at(0, 0, 0).transformation())));
    graph_holder holder(std::move(operands), {1, 2});
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
    REQUIRE(cells.size() == 2);
    for (auto &c : cells) {
      REQUIRE(tf::is_closed(c.polygons()));
      REQUIRE_THAT(std::abs(double(tf::signed_volume(c.polygons()))),
                   Catch::Matchers::WithinAbs(4.0, 1e-9));
    }
  };

  SECTION("same-orientation copy") { check_halves(plane); }
  SECTION("reversed copy") { check_halves(plane_rev); }
}

TEST_CASE("sheets: reversed coincident sheet keeps its own orientation",
          "[csg][sheets]") {
  // The duplicate is folded into the survivor's wall, but its bit must
  // still mean "behind ITS declared normal": with the copy reversed,
  // op(1) and op(2) select opposite halves.
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2));
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(4), sheets_real_t(4)); // z=0, +z
  sheets_mesh_t plane = tf::triangulated(planeq.polygons());
  sheets_mesh_t plane_rev = reversed_copy(plane);

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane_rev, tf::transformation<sheets_real_t, 3>(
                     sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1, 2});

  auto centroid_of_selection = [&](const tf::csg::expr &e) {
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph, e);
    REQUIRE(cells.size() == 1);
    REQUIRE_THAT(std::abs(double(tf::signed_volume(cells[0].polygons()))),
                 Catch::Matchers::WithinAbs(4.0, 1e-9));
    return centroid_z(cells[0]);
  };

  // Behind sheet 1 (+z normal) is below; behind the reversed copy is above.
  REQUIRE(centroid_of_selection(tf::csg::op(0) & tf::csg::op(1)) < 0.0);
  REQUIRE(centroid_of_selection(tf::csg::op(0) & tf::csg::op(2)) > 0.0);
}

TEST_CASE("sheets: three coincident sheets with mixed windings",
          "[csg][sheets]") {
  // Star-shaped folds: every dead loop folds onto one survivor with its
  // own reversed flag, so each tag anchors independently and the fused
  // outside clears all three bits.
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2));
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(4), sheets_real_t(4)); // z=0, +z
  sheets_mesh_t plane = tf::triangulated(planeq.polygons());
  sheets_mesh_t plane_rev = reversed_copy(plane);

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane_rev, tf::transformation<sheets_real_t, 3>(
                     sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1, 2, 3});

  auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
  REQUIRE(cells.size() == 2);

  auto half_behind = [&](int op_id) {
    auto [sel, sel_ids] = tf::test::csg_domains_of(
        holder.graph, tf::csg::op(0) & tf::csg::op(op_id));
    REQUIRE(sel.size() == 1);
    return centroid_z(sel[0]);
  };
  REQUIRE(half_behind(1) < 0.0); // behind +z normal = below
  REQUIRE(half_behind(2) > 0.0); // reversed copy: behind = above
  REQUIRE(half_behind(3) < 0.0); // forward copy: below again
}

TEST_CASE("sheets: reversed single sheet anchors behind its own normal",
          "[csg][sheets]") {
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2));
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(4), sheets_real_t(4)); // z=0, +z
  sheets_mesh_t plane_rev = reversed_copy(tf::triangulated(planeq.polygons()));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane_rev, tf::transformation<sheets_real_t, 3>(
                     sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {1});

  auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
  REQUIRE(cells.size() == 2);

  // Normal points -z, so behind the sheet is the upper half.
  auto [sel, sel_ids] =
      tf::test::csg_domains_of(holder.graph, tf::csg::op(0) & tf::csg::op(1));
  REQUIRE(sel.size() == 1);
  REQUIRE_THAT(std::abs(double(tf::signed_volume(sel[0].polygons()))),
               Catch::Matchers::WithinAbs(4.0, 1e-9));
  REQUIRE(centroid_z(sel[0]) > 0.0);
}

TEST_CASE("sheets: stack provenance follows orientation",
          "[csg][sheets][provenance]") {
  // Each half's wall is attributed to the sheet wound OUT of it: the +z
  // sheet below, the reversed sheet above. A same-orientation pair is
  // ambiguous per side, so the smallest tag carries both walls.
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2));
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(4), sheets_real_t(4)); // z=0, +z
  sheets_mesh_t plane = tf::triangulated(planeq.polygons());
  sheets_mesh_t plane_rev = reversed_copy(plane);

  SECTION("opposing pair splits the attribution") {
    std::vector<sheets_operand_t> operands;
    operands.push_back(tf::test::make_tagged_operand(
        box, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane_rev, tf::transformation<sheets_real_t, 3>(
                       sheets_frame_at(0, 0, 0).transformation())));
    graph_holder holder(std::move(operands), {1, 2});
    auto [cells, ids, imap] =
        tf::make_csg_domains(holder.graph, tf::return_index_map);
    REQUIRE(cells.size() == 2);
    for (std::size_t k = 0; k < cells.size(); ++k) {
      const sheets_index_t t =
          wall_tag_of(cells[k], [&imap = imap, k](std::size_t f) {
            return imap.face_tag_blocks[k][f];
          });
      REQUIRE(t == (centroid_z(cells[k]) < 0 ? sheets_index_t(1)
                                             : sheets_index_t(2)));
    }
    // the source-ids variant rides the same labels
    auto [scells, sids, tag_blocks, face_blocks] =
        tf::make_csg_domains(holder.graph, tf::return_source_ids);
    REQUIRE(scells.size() == 2);
    for (std::size_t k = 0; k < scells.size(); ++k) {
      const sheets_index_t t =
          wall_tag_of(scells[k], [&tag_blocks = tag_blocks, k](std::size_t f) {
            return tag_blocks[k][f];
          });
      REQUIRE(t == (centroid_z(scells[k]) < 0 ? sheets_index_t(1)
                                              : sheets_index_t(2)));
    }
  }

  SECTION("an aligned pair and an opposing member: each side its own") {
    // the side the aligned pair faces names the smaller of the pair; the
    // other side names the member wound out of it, never the pair
    std::vector<sheets_operand_t> operands;
    operands.push_back(tf::test::make_tagged_operand(
        box, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane_rev, tf::transformation<sheets_real_t, 3>(
                       sheets_frame_at(0, 0, 0).transformation())));
    graph_holder holder(std::move(operands), {1, 2, 3});
    auto [cells, ids, imap] =
        tf::make_csg_domains(holder.graph, tf::return_index_map);
    REQUIRE(cells.size() == 2);
    for (std::size_t k = 0; k < cells.size(); ++k) {
      const sheets_index_t t =
          wall_tag_of(cells[k], [&imap = imap, k](std::size_t f) {
            return imap.face_tag_blocks[k][f];
          });
      REQUIRE(t == (centroid_z(cells[k]) < 0 ? sheets_index_t(1)
                                             : sheets_index_t(3)));
    }
  }

  SECTION("same-orientation pair collapses to the smallest tag") {
    std::vector<sheets_operand_t> operands;
    operands.push_back(tf::test::make_tagged_operand(
        box, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    operands.push_back(tf::test::make_tagged_operand(
        plane, tf::transformation<sheets_real_t, 3>(
                   sheets_frame_at(0, 0, 0).transformation())));
    graph_holder holder(std::move(operands), {1, 2});
    auto [cells, ids, imap] =
        tf::make_csg_domains(holder.graph, tf::return_index_map);
    REQUIRE(cells.size() == 2);
    for (std::size_t k = 0; k < cells.size(); ++k) {
      const sheets_index_t t =
          wall_tag_of(cells[k], [&imap = imap, k](std::size_t f) {
            return imap.face_tag_blocks[k][f];
          });
      REQUIRE(t == sheets_index_t(1));
    }
  }
}

TEST_CASE("sheets: hole-cut coincident sheets keep sealed, attributed walls",
          "[csg][sheets][provenance]") {
  // A cylinder through the box AND both coincident opposing sheets: the
  // sheet faces around the hole carry non-simple (bridged) loops, the
  // exact shape a patched-in hole produces. Every domain must stay
  // closed and each z=0 wall (annulus and inner disk alike) must come
  // from the sheet wound out of its domain.
  auto boxq = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_real_t(2));
  sheets_mesh_t box = tf::triangulated(boxq.polygons());
  auto cylq = tf::make_cylinder_mesh<sheets_index_t>(sheets_real_t(0.5),
                                                     sheets_real_t(3), 24);
  sheets_mesh_t cyl = tf::triangulated(cylq.polygons());
  auto planeq =
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(4), sheets_real_t(4));
  sheets_mesh_t plane = tf::triangulated(planeq.polygons());
  sheets_mesh_t plane_rev = reversed_copy(plane);

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      cyl, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      plane_rev, tf::transformation<sheets_real_t, 3>(
                     sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {2, 3});
  auto [cells, ids, imap] =
      tf::make_csg_domains(holder.graph, tf::return_index_map);
  REQUIRE(cells.size() == 6);

  int walled = 0;
  for (std::size_t k = 0; k < cells.size(); ++k) {
    REQUIRE(tf::is_closed(cells[k].polygons()));
    const sheets_index_t t =
        wall_tag_of(cells[k], [&imap = imap, k](std::size_t f) {
          return imap.face_tag_blocks[k][f];
        });
    if (t == -2)
      continue; // cylinder stub outside the box: no z=0 wall
    ++walled;
    REQUIRE(t ==
            (centroid_z(cells[k]) < 0 ? sheets_index_t(2) : sheets_index_t(3)));
  }
  REQUIRE(walled == 4);
}

// ============================================================================
// The measured matrix: a sheet plane against a box. Sheet = tag 0.
// ============================================================================
TEST_CASE("sheets: plane and box, every read", "[csg][sheets]") {
  using tf::csg::inside;
  using tf::csg::op;
  using tf::csg::selection;
  auto plane = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_index_t(2), sheets_index_t(2));
  auto box = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(1), sheets_real_t(1), sheets_real_t(1));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      plane, tf::transformation<sheets_real_t, 3>(
                 sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {0});
  auto &graph = holder.graph;

  SECTION("regions bounded by the box and the sheet are closed, capped halves") {
    auto upper = tf::test::csg_mesh_of(graph, ~op(0) & op(1));
    check_closed_solid(upper, 0.5, 1e-12);
    REQUIRE(upper.size() == 22);
    REQUIRE(centroid_z(upper) > 0.0);
    auto lower = tf::test::csg_mesh_of(graph, op(0) & op(1));
    check_closed_solid(lower, 0.5, 1e-12);
    REQUIRE(lower.size() == 22);
    REQUIRE(centroid_z(lower) < 0.0);
    REQUIRE(tf::test::csg_mesh_of(graph, op(1) - op(0)).size() == 22);
  }
  SECTION("regions bounded by the sheet's half-space are open along its rim") {
    for (auto e :
         {~op(0) & ~op(1), op(0) & ~op(1), op(0) | op(1), op(0) - op(1)}) {
      auto m = tf::test::csg_mesh_of(graph, e);
      REQUIRE(m.size() == 30);
      REQUIRE(n_boundary(m) == 8);
      REQUIRE_FALSE(tf::is_closed(m.polygons()));
    }
  }
  SECTION("the boundary read of the sheet needs the sheet's own side") {
    REQUIRE(
        tf::test::csg_mesh_of(graph, selection({0}, ~op(0) & ~op(1))).size() ==
        16);
    REQUIRE(
        tf::test::csg_mesh_of(graph, selection({0}, ~op(0) & op(1))).size() ==
        8);
    REQUIRE(
        tf::test::csg_mesh_of(graph, selection({0}, op(0) & op(1))).size() ==
        8);
    REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, op(1))).size() == 0);
  }
  SECTION("inside: the sheet inside the box is the cap, stored winding") {
    auto cap = tf::test::csg_mesh_of(graph, inside({0}, op(1)));
    REQUIRE(cap.size() == 8);
    REQUIRE(n_boundary(cap) == 8);
    REQUIRE(faces_point(cap, 2, +1.0));
    auto idiom = tf::test::csg_mesh_of(graph, selection({0}, op(0) & op(1)));
    REQUIRE(idiom.size() == 8);
    REQUIRE(faces_point(idiom, 2, +1.0));
    auto twin = tf::test::csg_mesh_of(graph, selection({0}, ~op(0) & op(1)));
    REQUIRE(faces_point(twin, 2, -1.0));
  }
  SECTION("inside: the sheet outside the box is the annulus") {
    auto annulus = tf::test::csg_mesh_of(graph, inside({0}, ~op(1)));
    REQUIRE(annulus.size() == 16);
    REQUIRE(n_boundary(annulus) == 16);
    REQUIRE(faces_point(annulus, 2, +1.0));
  }
  SECTION("inside: the box's walls behind the sheet, outward") {
    auto walls = tf::test::csg_mesh_of(graph, inside({1}, op(0)));
    REQUIRE(walls.size() == 14);
    REQUIRE(centroid_z(walls) < 0.0);
    REQUIRE_FALSE(tf::is_closed(walls.polygons()));
  }
  SECTION("inside: a form's own bit is never on both sides") {
    REQUIRE(tf::test::csg_mesh_of(graph, inside({0}, op(0))).size() == 0);
    REQUIRE(tf::test::csg_mesh_of(graph, inside({0}, op(0) & op(1))).size() ==
            0);
    REQUIRE(tf::test::csg_mesh_of(graph, inside({1}, op(1))).size() == 0);
    REQUIRE(tf::test::csg_mesh_of(graph, inside({0}, op(0) | op(1))).size() ==
            8);
  }
  SECTION("inside carries provenance and the index map like any read") {
    auto [m, tags, faces] =
        tf::test::csg_mesh_with_source_ids_of(graph, inside({0}, op(1)));
    REQUIRE(m.size() == 8);
    REQUIRE(tags.size() == 8);
    for (auto t : tags)
      REQUIRE(t == 0);
    REQUIRE(faces.size() == 8);
    auto [mm, imap] =
        tf::test::csg_mesh_with_index_map_of(graph, inside({0}, op(1)));
    REQUIRE(mm.size() == 8);
    REQUIRE(imap.face_labels.size() == 8);
  }
  SECTION("domains: the default keeps the two capped halves") {
    auto [cells, ids] = tf::test::csg_domains_of(graph);
    REQUIRE(cells.size() == 2);
    for (auto &c : cells)
      check_closed_solid(c, 0.5, 1e-12);
    auto [raw, raw_ids] =
        tf::test::csg_domains_of(graph, tf::domain_config::none);
    REQUIRE(raw.size() == 4);
    auto [shelled, shelled_ids] =
        tf::test::csg_domains_of(graph, tf::domain_config::exclude_outer_shell);
    REQUIRE(shelled.size() == 3);
  }
}

// ============================================================================
// Two crossing sheets carve four wedges. A: z = 0, +z. B: y = 0, normal -y,
// so op(1) is the y > 0 half-space.
// ============================================================================
TEST_CASE("sheets: two crossing planes carve four wedges", "[csg][sheets]") {
  using tf::csg::inside;
  using tf::csg::op;
  using tf::csg::selection;
  auto a = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_index_t(2), sheets_index_t(2));
  auto b = rotated_about_x(tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(2), sheets_real_t(2), sheets_index_t(2),
      sheets_index_t(2)));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      a, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      b, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {0, 1});
  auto &graph = holder.graph;

  SECTION("every wedge's boundary is two half-planes, open") {
    for (auto e : {~op(0) & ~op(1), ~op(0) & op(1), op(0) & ~op(1),
                   op(0) & op(1), op(0) | op(1), op(0) - op(1),
                   op(1) - op(0)}) {
      auto m = tf::test::csg_mesh_of(graph, e);
      REQUIRE(m.size() == 8);
      REQUIRE(n_boundary(m) == 8);
      REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, e)).size() == 4);
    }
    REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, op(1))).size() == 0);
  }
  SECTION("inside: A's half behind B") {
    auto half = tf::test::csg_mesh_of(graph, inside({0}, op(1)));
    REQUIRE(half.size() == 4);
    REQUIRE(n_boundary(half) == 6);
    REQUIRE(centroid_y(half) > 0.0);
    REQUIRE(faces_point(half, 2, +1.0));
    auto other = tf::test::csg_mesh_of(graph, inside({0}, ~op(1)));
    REQUIRE(other.size() == 4);
    REQUIRE(centroid_y(other) < 0.0);
    auto b_half = tf::test::csg_mesh_of(graph, inside({1}, op(0)));
    REQUIRE(b_half.size() == 4);
    REQUIRE(centroid_z(b_half) < 0.0);
    REQUIRE(faces_point(b_half, 1, -1.0));
  }
  SECTION("domains: the default fuses every wedge away; none keeps four") {
    auto [cells, ids] = tf::test::csg_domains_of(graph);
    REQUIRE(cells.size() == 0);
    auto [wedges, wedge_ids] =
        tf::test::csg_domains_of(graph, tf::domain_config::none);
    REQUIRE(wedges.size() == 4);
    for (auto &w : wedges) {
      REQUIRE(w.size() == 8);
      REQUIRE_FALSE(tf::is_closed(w.polygons()));
    }
    auto [three, three_ids] =
        tf::test::csg_domains_of(graph, tf::domain_config::exclude_outer_shell);
    REQUIRE(three.size() == 3);
  }
}

// ============================================================================
// A flap: a sheet wholly inside a volume separates nothing. No boundary read
// sees it, and inside sees it once, embedded in the region that holds it.
// ============================================================================
TEST_CASE("sheets: a flap inside a volume is inside it once", "[csg][sheets]") {
  using tf::csg::inside;
  using tf::csg::op;
  using tf::csg::selection;
  auto flap = tf::make_plane_mesh<sheets_index_t>(
      sheets_real_t(0.5), sheets_real_t(0.5), sheets_index_t(1),
      sheets_index_t(1));
  auto box = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(1), sheets_real_t(1), sheets_real_t(1));

  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      flap, tf::transformation<sheets_real_t, 3>(
                sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      box, tf::transformation<sheets_real_t, 3>(
               sheets_frame_at(0, 0, 0).transformation())));
  graph_holder holder(std::move(operands), {0});
  auto &graph = holder.graph;

  REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, ~op(0) & op(1))).size() ==
          0);
  REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, op(0) & op(1))).size() ==
          0);
  REQUIRE(tf::test::csg_mesh_of(graph, selection({0}, op(1))).size() == 0);
  auto once = tf::test::csg_mesh_of(graph, inside({0}, op(1)));
  REQUIRE(once.size() == 2);
  REQUIRE(n_boundary(once) == 4);
  REQUIRE(faces_point(once, 2, +1.0));
  REQUIRE(tf::test::csg_mesh_of(graph, inside({0}, ~op(0) & op(1))).size() ==
          2);
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  REQUIRE(cells.size() == 1);
  check_closed_solid(cells[0], 1.0, 1e-12);
}

// ============================================================================
// inside on a volume: B's shell inside A.
// ============================================================================
TEST_CASE("sheets: inside reads a volume's shell within another",
          "[csg][sheets]") {
  using tf::csg::inside;
  using tf::csg::op;
  using tf::csg::selection;
  auto a = tf::make_box_mesh<sheets_index_t>(sheets_real_t(1), sheets_real_t(1),
                                             sheets_real_t(1));
  auto b = shifted_x(tf::make_box_mesh<sheets_index_t>(
                         sheets_real_t(1), sheets_real_t(1), sheets_real_t(1)),
                     sheets_real_t(0.5));
  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      a, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      b, tf::transformation<sheets_real_t, 3>(
             sheets_frame_at(0, 0, 0).transformation())));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto in_a = tf::test::csg_mesh_of(graph, inside({1}, op(0)));
  REQUIRE(in_a.size() == 6);
  REQUIRE(in_a.size() ==
          tf::test::csg_mesh_of(graph, selection({1}, op(0) & op(1))).size());
  REQUIRE(tf::test::csg_mesh_of(graph, inside({1}, op(1))).size() == 0);
  REQUIRE(tf::test::csg_mesh_of(graph, inside({0}, op(0))).size() == 0);
  REQUIRE(tf::test::csg_mesh_of(graph, inside({1}, ~op(0))).size() ==
          tf::test::csg_mesh_of(graph, selection({1}, ~op(0) & op(1))).size());
  REQUIRE(tf::test::csg_mesh_of(graph, inside({1}, ~op(0))).size() == 14);
  REQUIRE(tf::test::csg_mesh_of(graph, selection({1}, op(0))).size() == 0);
}

// ============================================================================
// A sheet is open only where its rim meets nothing. The box [-5,5]^3 and a
// square sheet on z = 0: inside the box it floats and divides nothing,
// flush with the walls its rim is welded onto them and it severs the box,
// and overhanging it severs the box as well. A rim crossing the box's
// interior between the two walls it pierces meets nothing and stays open.
// ============================================================================
namespace {

auto sheets_rim_scene(sheets_real_t width, sheets_real_t depth)
    -> std::vector<sheets_operand_t> {
  std::vector<sheets_operand_t> operands;
  operands.push_back(tf::test::make_tagged_operand(
      tf::make_box_mesh<sheets_index_t>(sheets_real_t(10), sheets_real_t(10),
                                        sheets_real_t(10)),
      tf::transformation<sheets_real_t, 3>(
          sheets_frame_at(0, 0, 0).transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      tf::make_plane_mesh<sheets_index_t>(width, depth),
      tf::transformation<sheets_real_t, 3>(
          sheets_frame_at(0, 0, 0).transformation())));
  return operands;
}

/// Whether every component of `tag` is open.
template <typename Graph>
auto sheets_tag_open(const Graph &graph, sheets_index_t tag) -> bool {
  const auto &desc = graph.descriptor();
  auto open_mask = graph.labels().open_component_mask();
  bool any = false;
  for (std::size_t c = 0; c < open_mask.size(); ++c) {
    if (desc.tag_of_component[c] != tag)
      continue;
    any = true;
    if (!open_mask[c])
      return false;
  }
  return any;
}

/// The volumes of the cells that enclose something, ascending; every cell
/// is a closed manifold. Whether the unbounded outside is among the cells is
/// the exclusion's question, not the rim's.
template <typename Cells>
auto sheets_enclosed_volumes(const Cells &cells) -> std::vector<double> {
  std::vector<double> volumes;
  for (const auto &cell : cells) {
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(tf::is_manifold(cell.polygons()));
    const double volume = sheets_volume_of(cell);
    if (volume > 0.0)
      volumes.push_back(volume);
  }
  std::sort(volumes.begin(), volumes.end());
  return volumes;
}

void sheets_volumes_check(const std::vector<double> &volumes,
                          const std::vector<double> &expected) {
  REQUIRE(volumes.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    REQUIRE_THAT(volumes[i], Catch::Matchers::WithinAbs(expected[i], 1e-9));
}

} // namespace

TEST_CASE("sheets: the exclusion drops the universe the volumes name",
          "[csg][sheets]") {
  graph_holder holder(sheets_rim_scene(10, 10), {1});
  REQUIRE(holder.graph.failed().size() == 0);
  SECTION("no flags: the universe is a kept cell") {
    auto [cells, ids] =
        tf::test::csg_domains_of(holder.graph, tf::domain_config::none);
    REQUIRE(cells.size() == 3);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
  SECTION("exclude_outer_shell: only the halves remain") {
    auto [cells, ids] = tf::test::csg_domains_of(
        holder.graph, tf::domain_config::exclude_outer_shell);
    REQUIRE(cells.size() == 2);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
  SECTION("both flags: only the halves remain") {
    auto [cells, ids] = tf::test::csg_domains_of(
        holder.graph, tf::domain_config::exclude_outer_shell |
                          tf::domain_config::ignore_open_fragments);
    REQUIRE(cells.size() == 2);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
}

TEST_CASE("sheets: a rim is open where it meets nothing", "[csg][sheets]") {
  SECTION("floating inside the box: one domain") {
    graph_holder holder(sheets_rim_scene(8, 8), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE(sheets_tag_open(holder.graph, 1));
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {1000.0});
  }
  SECTION("flush with the walls: welded, closed, two halves") {
    graph_holder holder(sheets_rim_scene(10, 10), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE_FALSE(sheets_tag_open(holder.graph, 1));
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
  SECTION("flush with the walls, open fragments fused: still two halves") {
    graph_holder holder(sheets_rim_scene(10, 10), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    auto [cells, ids] = tf::test::csg_domains_of(
        holder.graph, tf::domain_config::ignore_open_fragments);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
  SECTION("overhanging the walls: two halves") {
    graph_holder holder(sheets_rim_scene(11, 11), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {500.0, 500.0});
  }
  SECTION("a rim across the interior between two created points stays open") {
    graph_holder holder(sheets_rim_scene(16, 6), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE(sheets_tag_open(holder.graph, 1));
    auto [cells, ids] = tf::test::csg_domains_of(holder.graph);
    sheets_volumes_check(sheets_enclosed_volumes(cells), {1000.0});
  }
}

// ============================================================================
// A sheet has no outside, so the outside the forms stand in takes each
// sheet's side from the sheet's winding field. A sheet floating above the
// box [-5,5]^3 never touches it, so the whole box lies on one side of it,
// and the box's own interior reaches that side through the outside.
// ============================================================================
namespace {

auto shifted_z(sheets_mesh_t m, sheets_real_t dz) -> sheets_mesh_t {
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i)
    m.points_buffer()[i][2] += dz;
  return m;
}

auto sheets_identity_operand(sheets_mesh_t m) -> sheets_operand_t {
  return tf::test::make_tagged_operand(
      std::move(m), tf::transformation<sheets_real_t, 3>(
                        sheets_frame_at(0, 0, 0).transformation()));
}

} // namespace

TEST_CASE("sheets: the outside takes a floating sheet's side",
          "[csg][sheets]") {
  using tf::csg::op;
  using tf::csg::selection;
  auto box = tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(10), sheets_real_t(10), sheets_real_t(10));
  auto sheet = shifted_z(
      tf::make_plane_mesh<sheets_index_t>(sheets_real_t(2), sheets_real_t(2)),
      sheets_real_t(8));
  SECTION("the box lies behind the sheet") {
    std::vector<sheets_operand_t> operands;
    operands.push_back(sheets_identity_operand(box));
    operands.push_back(sheets_identity_operand(sheet));
    graph_holder holder(std::move(operands), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(
        tf::test::csg_mesh_of(holder.graph, selection({0}, op(0) & op(1))),
        1000.0, 1e-9);
    REQUIRE(tf::test::csg_mesh_of(holder.graph, selection({0}, op(0) & ~op(1)))
                .size() == 0);
  }
  SECTION("the box lies in front of the sheet") {
    std::vector<sheets_operand_t> operands;
    operands.push_back(sheets_identity_operand(box));
    operands.push_back(sheets_identity_operand(
        shifted_z(rotated_about_x(rotated_about_x(shifted_z(sheet, -8))), 8)));
    graph_holder holder(std::move(operands), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE(tf::test::csg_mesh_of(holder.graph, selection({0}, op(0) & op(1)))
                .size() == 0);
    check_closed_solid(
        tf::test::csg_mesh_of(holder.graph, selection({0}, op(0) & ~op(1))),
        1000.0, 1e-9);
  }
}

// ============================================================================
// A box of [-5,5]^3 holds two sheets on z = 0 and on that plane turned 15
// degrees about y; the turned one outruns the flat one along their crossing,
// so the outside air goes around the flat sheet's end and states no side of
// it. Every cell bounds both sheets, so its walls state its row, and every
// cell lies wholly on one side of each sheet's plane, so its centroid states
// the same row.
// ============================================================================
namespace {

auto sheets_turned_about_y(sheets_mesh_t m, double degrees) -> sheets_mesh_t {
  tf::transform(m.polygons(), tf::make_rotation(tf::deg(sheets_real_t(degrees)),
                                                tf::axis<1>));
  return m;
}

auto sheets_unit_normal(const sheets_mesh_t &m) -> std::array<double, 3> {
  auto pts = m.polygons().points();
  auto face = m.faces()[0];
  const auto n =
      tf::cross(pts[face[1]] - pts[face[0]], pts[face[2]] - pts[face[0]]);
  const double l =
      std::sqrt(double(n[0]) * double(n[0]) + double(n[1]) * double(n[1]) +
                double(n[2]) * double(n[2]));
  return {double(n[0]) / l, double(n[1]) / l, double(n[2]) / l};
}

} // namespace

TEST_CASE("sheets: crossing sheets state every cell's side by its walls",
          "[csg][sheets]") {
  auto flat = tf::make_plane_mesh<sheets_index_t>(sheets_real_t(11),
                                                  sheets_real_t(11), 1, 1);
  auto turned =
      sheets_turned_about_y(tf::make_plane_mesh<sheets_index_t>(
                                sheets_real_t(11), sheets_real_t(12), 1, 1),
                            15.0);
  const std::array<std::array<double, 3>, 2> normals{
      sheets_unit_normal(flat), sheets_unit_normal(turned)};
  std::vector<sheets_operand_t> operands;
  operands.push_back(sheets_identity_operand(tf::make_box_mesh<sheets_index_t>(
      sheets_real_t(10), sheets_real_t(10), sheets_real_t(10), 1, 1, 1)));
  operands.push_back(sheets_identity_operand(std::move(flat)));
  operands.push_back(sheets_identity_operand(std::move(turned)));
  graph_holder holder(std::move(operands), {1, 2});
  REQUIRE(holder.graph.failed().size() == 0);

  const auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(
      holder.graph, tf::domain_config::ignore_open_fragments |
                        tf::domain_config::exclude_outer_shell);
  REQUIRE(cells.size() == 4);
  for (std::size_t k = 0; k < cells.size(); ++k) {
    std::array<double, 3> c{0.0, 0.0, 0.0};
    const auto pts = cells[k].points();
    for (const auto &p : pts)
      for (int i = 0; i < 3; ++i)
        c[i] += double(p[i]) / double(pts.size());
    REQUIRE(imap.inclusion[k][0]);
    for (std::size_t s = 0; s < 2; ++s) {
      const auto &n = normals[s];
      const bool behind = c[0] * n[0] + c[1] * n[1] + c[2] * n[2] < 0.0;
      REQUIRE(imap.inclusion[k][s + 1] == behind);
    }
  }
}

// ============================================================================
// Every index-map axis of the domains runs parallel to the cells, so a kept
// cell that received no face still owns its blocks. Two crossing sheets
// alone are all open fragments, and fusing them leaves one empty cell.
// ============================================================================
TEST_CASE("sheets: fused open fragments leave one block per cell",
          "[csg][sheets]") {
  std::vector<sheets_operand_t> operands;
  operands.push_back(
      sheets_identity_operand(tf::make_plane_mesh<sheets_index_t>(
          sheets_real_t(11), sheets_real_t(11), 5, 5)));
  operands.push_back(sheets_identity_operand(
      sheets_turned_about_y(tf::make_plane_mesh<sheets_index_t>(
                                sheets_real_t(11), sheets_real_t(11), 5, 5),
                            15.0)));
  graph_holder holder(std::move(operands), {0, 1});
  REQUIRE(holder.graph.failed().size() == 0);
  const auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(
      holder.graph, tf::domain_config::ignore_open_fragments);
  REQUIRE(cells.size() == 1);
  REQUIRE(ids.size() == cells.size());
  REQUIRE(std::size_t(imap.inclusion.size()) == cells.size());
  REQUIRE(imap.face_tag_blocks.size() == cells.size());
  REQUIRE(imap.face_blocks.size() == cells.size());
  REQUIRE(imap.point_tag_blocks.size() == cells.size());
  REQUIRE(imap.point_blocks.size() == cells.size());
  for (std::size_t k = 0; k < cells.size(); ++k) {
    REQUIRE(imap.face_tag_blocks[k].size() == cells[k].faces().size());
    REQUIRE(imap.face_blocks[k].size() == cells[k].faces().size());
    REQUIRE(imap.point_tag_blocks[k].size() == cells[k].points().size());
    REQUIRE(imap.point_blocks[k].size() == cells[k].points().size());
  }
}

// ============================================================================
// A sheet on z = 2 runs through the box [-5,5]^3 and ends at |x|,|y| = 7
// inside the closed box T = [-9,9]^2 x [-9,3]. Slabs stop inside the box
// below the sheet and either share T's sides and floor or poke through an
// outer box around everything, so they carve the cells around the rim and
// the air reaches the box's floor without passing the cell that holds the
// rim. The box is what the sheet separates, so its cells keep the sides its
// walls state however many slabs there are.
// ============================================================================
namespace {

auto sheets_law_box(sheets_real_t x0, sheets_real_t y0, sheets_real_t z0,
                    sheets_real_t x1, sheets_real_t y1, sheets_real_t z1)
    -> sheets_mesh_t {
  auto m =
      tf::make_box_mesh<sheets_index_t>(x1 - x0, y1 - y0, z1 - z0);
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i) {
    m.points_buffer()[i][0] += (x0 + x1) / 2;
    m.points_buffer()[i][1] += (y0 + y1) / 2;
    m.points_buffer()[i][2] += (z0 + z1) / 2;
  }
  return m;
}

auto sheets_law_cube(sheets_real_t lo, sheets_real_t hi, sheets_real_t z0,
                     sheets_real_t z1) -> sheets_operand_t {
  return sheets_identity_operand(sheets_law_box(lo, lo, z0, hi, hi, z1));
}

// the plane z = `z` over [x0, x1] x [y0, y1], its normal +z
auto sheets_law_plane(sheets_real_t x0, sheets_real_t y0, sheets_real_t x1,
                      sheets_real_t y1, sheets_real_t z) -> sheets_mesh_t {
  auto m = tf::make_plane_mesh<sheets_index_t>(x1 - x0, y1 - y0, 1, 1);
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i) {
    m.points_buffer()[i][0] += (x0 + x1) / 2;
    m.points_buffer()[i][1] += (y0 + y1) / 2;
    m.points_buffer()[i][2] += z;
  }
  return m;
}

auto sheets_enclosed_rim_scene(int n_slabs, sheets_real_t slab_side,
                               bool rim_box, bool outer_box = false)
    -> std::vector<sheets_operand_t> {
  std::vector<sheets_operand_t> operands;
  operands.push_back(sheets_law_cube(-5, 5, -5, 5));
  operands.push_back(
      sheets_identity_operand(sheets_law_plane(-7, -7, 7, 7, 2)));
  for (int k = 0; k < n_slabs; ++k)
    operands.push_back(
        sheets_law_cube(-slab_side, slab_side, -slab_side,
                        sheets_real_t(-4) + sheets_real_t(k) / 2));
  if (rim_box)
    operands.push_back(sheets_law_cube(-9, 9, -9, 3));
  if (outer_box)
    operands.push_back(sheets_law_cube(-12, 12, -12, 12));
  return operands;
}

// the volumes that hold a sheet's free rim, as (sheet tag, volume tag)
void sheets_rim_census(const graph_holder &holder,
                       std::vector<std::array<int, 2>> expected) {
  std::vector<std::array<int, 2>> got;
  for (const auto &pair : holder.graph.sheet_rim_volumes())
    got.push_back({int(pair[0]), int(pair[1])});
  REQUIRE(got == expected);
}

void sheets_check_box_halves(const graph_holder &holder) {
  using tf::csg::op;
  REQUIRE(holder.graph.failed().size() == 0);
  REQUIRE(holder.graph.n_sheet_conflicts() == 0);
  check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(1)), 700.0,
                     1e-9);
  check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)), 300.0,
                     1e-9);
}

} // namespace

TEST_CASE("sheets: a free rim closed in by other forms' walls",
          "[csg][sheets]") {
  for (const int n : {1, 5, 10}) {
    DYNAMIC_SECTION("slabs sharing the rim's box, n = " << n) {
      graph_holder holder(sheets_enclosed_rim_scene(n, 9, true), {1});
      sheets_check_box_halves(holder);
      sheets_rim_census(holder, {{1, n + 2}});
    }
    DYNAMIC_SECTION("slabs poking through an outer box, n = " << n) {
      graph_holder holder(sheets_enclosed_rim_scene(n, 13, true, true), {1});
      sheets_check_box_halves(holder);
      sheets_rim_census(holder, {{1, n + 2}, {1, n + 3}});
    }
  }
  SECTION("slabs inside the rim's box") {
    graph_holder holder(sheets_enclosed_rim_scene(5, sheets_real_t(8.5), true),
                        {1});
    sheets_check_box_halves(holder);
    sheets_rim_census(holder, {{1, 7}});
  }
  SECTION("no box around the rim") {
    graph_holder holder(sheets_enclosed_rim_scene(5, 9, false), {1});
    sheets_check_box_halves(holder);
    sheets_rim_census(holder, {});
  }
}

// ============================================================================
// The law of sheet sides (csg/graph/sheet_sides.hpp), one clause per
// section. A volume the sheet's walls reach from one side lies on it, one
// they reach from both carries no side, one they never reach falls through
// to the winding, and a one-side statement outranks a wrap.
// ============================================================================
namespace {

auto sheets_law_operands(std::vector<sheets_mesh_t> meshes)
    -> std::vector<sheets_operand_t> {
  std::vector<sheets_operand_t> operands;
  for (auto &m : meshes)
    operands.push_back(sheets_identity_operand(std::move(m)));
  return operands;
}

auto sheets_law_joined(const sheets_mesh_t &a, const sheets_mesh_t &b)
    -> sheets_mesh_t {
  sheets_mesh_t m;
  const auto na = sheets_index_t(a.points_buffer().size());
  m.faces_buffer().allocate(a.faces().size() + b.faces().size());
  m.points_buffer().allocate(a.points_buffer().size() +
                             b.points_buffer().size());
  std::size_t f = 0;
  for (auto face : a.faces()) {
    for (int j = 0; j < 3; ++j)
      m.faces_buffer()[f][j] = face[j];
    ++f;
  }
  for (auto face : b.faces()) {
    for (int j = 0; j < 3; ++j)
      m.faces_buffer()[f][j] = face[j] + na;
    ++f;
  }
  std::size_t p = 0;
  for (auto q : a.points())
    m.points_buffer()[p++] = q;
  for (auto q : b.points())
    m.points_buffer()[p++] = q;
  return m;
}

// The U of the (x, z) outline (0,0) (6,0) (6,8) (4,8) (4,2) (2,2) (2,8)
// (0,8), extruded over y in [0, 2]: two arms x in [0,2] and [4,6] over a
// base z in [0,2]. Volume 72.
auto sheets_law_u() -> sheets_mesh_t {
  const std::array<std::array<sheets_real_t, 2>, 8> outline{
      {{0, 0}, {6, 0}, {6, 8}, {4, 8}, {4, 2}, {2, 2}, {2, 8}, {0, 8}}};
  const std::array<std::array<sheets_index_t, 3>, 6> cap{
      {{0, 1, 4}, {1, 2, 3}, {1, 3, 4}, {0, 4, 5}, {0, 5, 6}, {0, 6, 7}}};
  sheets_mesh_t m;
  m.points_buffer().allocate(16);
  for (std::size_t i = 0; i < 8; ++i) {
    m.points_buffer()[i] =
        tf::point<sheets_real_t, 3>{outline[i][0], 0, outline[i][1]};
    m.points_buffer()[i + 8] =
        tf::point<sheets_real_t, 3>{outline[i][0], 2, outline[i][1]};
  }
  m.faces_buffer().allocate(2 * cap.size() + 16);
  std::size_t f = 0;
  auto put = [&](sheets_index_t a, sheets_index_t b, sheets_index_t c) {
    m.faces_buffer()[f][0] = a;
    m.faces_buffer()[f][1] = b;
    m.faces_buffer()[f][2] = c;
    ++f;
  };
  for (const auto &t : cap) {
    put(t[0], t[1], t[2]);
    put(t[0] + 8, t[2] + 8, t[1] + 8);
  }
  for (sheets_index_t i = 0; i < 8; ++i) {
    const sheets_index_t j = (i + 1) % 8;
    put(j, i, i + 8);
    put(j, i + 8, j + 8);
  }
  tf::ensure_positive_orientation(m.polygons());
  return m;
}

auto sheets_law_volume(const graph_holder &holder, tf::csg::expr e) -> double {
  return sheets_volume_of(tf::test::csg_mesh_of(holder.graph, e));
}

} // namespace

TEST_CASE("sheets: the law of sheet sides, clause by clause",
          "[csg][sheets]") {
  using tf::csg::op;
  SECTION("one side and both sides: a box the sheet crosses, a box it ends "
          "in") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -2, -2, -1, 2, 2),
                             sheets_law_box(1, -2, -2, 5, 2, 2),
                             sheets_law_plane(-6, -3, 3, 3, 0)}),
        {2});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(2)),
                       32.0, 1e-9);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(2)),
                       32.0, 1e-9);
    REQUIRE(tf::test::csg_mesh_of(holder.graph, op(1) & op(2)).size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(1) - op(2)),
                       64.0, 1e-9);
  }
  SECTION("the component, not the volume: an arm cut off, the rest wrapped") {
    auto sheet = reversed_copy(sheets_law_plane(-1, -1, 5, 3, 6));
    graph_holder holder(sheets_law_operands({sheets_law_u(), sheet}), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(1)), 8.0,
                       1e-9);
    REQUIRE_THAT(sheets_law_volume(holder, op(0) - op(1)),
                 Catch::Matchers::WithinAbs(64.0, 1e-9));
  }
  SECTION("never reached: a volume's component the sheet misses takes the "
          "winding") {
    graph_holder holder(
        sheets_law_operands(
            {sheets_law_joined(sheets_law_box(-5, -5, -5, 5, 5, 5),
                               sheets_law_box(-2, -2, -12, 2, 2, -8)),
             sheets_law_plane(-7, -7, 7, 7, 0)}),
        {1});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE_THAT(sheets_law_volume(holder, op(0) & op(1)),
                 Catch::Matchers::WithinAbs(564.0, 1e-9));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       500.0, 1e-9);
  }
  SECTION("one side outranks a wrap where two volumes overlap") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_box(3, -9, -2, 9, 9, 2),
                             sheets_law_plane(-7, -7, 7, 7, 0)}),
        {2});
    REQUIRE(holder.graph.failed().size() == 0);
    REQUIRE(holder.graph.n_sheet_conflicts() == 0);
    check_closed_solid(
        tf::test::csg_mesh_of(holder.graph, op(0) & op(1) & op(2)), 40.0,
        1e-9);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(2)),
                       500.0, 1e-9);
    REQUIRE(tf::test::csg_mesh_of(holder.graph, (op(1) - op(0)) & op(2))
                .size() == 0);
  }
  SECTION("floaters the sheet never reaches: under its middle, and beside "
          "its rim in its own plane") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-1, -1, -12, 1, 1, -10),
                             sheets_law_box(8, -1, -1, 10, 1, 1),
                             sheets_law_plane(-7, -7, 7, 7, 0)}),
        {2});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(2)), 8.0,
                       1e-9);
    const double beside = sheets_law_volume(holder, op(1) & op(2));
    REQUIRE((std::abs(beside) < 1e-9 || std::abs(beside - 8.0) < 1e-9));
  }
  SECTION("stacked rims: two sheets on one free rim, both separating") {
    auto operands = sheets_enclosed_rim_scene(5, 9, true);
    operands.insert(operands.begin() + 2,
                    sheets_identity_operand(sheets_law_plane(-7, -7, 7, 7, 2)));
    graph_holder holder(std::move(operands), {1, 2});
    REQUIRE(holder.graph.failed().size() == 0);
    for (const int s : {1, 2}) {
      check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(s)),
                         700.0, 1e-9);
      check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(s)),
                         300.0, 1e-9);
    }
  }
  SECTION("a rim on the volume's own walls separates it") {
    graph_holder holder(sheets_rim_scene(10, 10), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(1)),
                       500.0, 1e-9);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       500.0, 1e-9);
  }
  SECTION("a mirrored sheet keeps its sides on both winding paths") {
    // z -> -z keeps the plane on z = 0 and turns its normal back onto +z,
    // so the floater below it lies behind it
    tf::transformation<sheets_real_t, 3> mirror;
    mirror.fill({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0});
    std::vector<sheets_operand_t> operands;
    operands.push_back(
        sheets_identity_operand(sheets_law_box(-1, -1, -4, 1, 1, -2)));
    operands.push_back(tf::test::make_tagged_operand(
        sheets_law_plane(-7, -7, 7, 7, 0), mirror));
    graph_holder holder(std::move(operands), {1});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(1)), 8.0,
                       1e-9);
    REQUIRE(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)).size() == 0);

    // one query takes the pass on the lattice, a batch the folded moments
    using int_t =
        typename std::decay_t<decltype(holder.graph)>::resolved_int_type;
    using corners_t = std::array<tf::exact::pt3<int_t>, 3>;
    const auto &arrangement = holder.graph.arrangement();
    auto reader = arrangement.lattice().reader(arrangement.apply_to_form());
    auto centroid_at = [&](sheets_real_t z) -> corners_t {
      using pt_t = tf::point<sheets_real_t, 3>;
      return {reader.converter.convert(pt_t{-0.5, -0.5, z}),
              reader.converter.convert(pt_t{0.5, -0.5, z}),
              reader.converter.convert(pt_t{0, 0.5, z})};
    };
    tf::buffer<corners_t> one;
    one.push_back(centroid_at(-3));
    tf::buffer<corners_t> batch;
    batch.push_back(centroid_at(-3));
    while (batch.size() < tf::csg::graph::winding_moments_from)
      batch.push_back(centroid_at(3));
    arrangement.apply_to_form()(sheets_index_t(1), [&](const auto &form) {
      const auto linear = tf::csg::graph::winding_side<int_t>(
          form, sheets_index_t(1), reader, one);
      const auto folded = tf::csg::graph::winding_side<int_t>(
          form, sheets_index_t(1), reader, batch);
      REQUIRE(linear[0] == char(1));
      REQUIRE(folded[0] == char(1));
      for (std::size_t j = 1; j < folded.size(); ++j)
        REQUIRE(folded[j] == char(0));
    });
  }
}

// ============================================================================
// A free rim joins the two sides of its sheet, so every cell a volume's
// inside reaches around the rim reads no side of the sheet, unless a volume
// the sheet separates states one. A component cut at a volume's wall has no
// free rim, so a volume the sheet crosses edge to edge keeps its sides.
// ============================================================================
namespace {

void sheets_rim_empty(const graph_holder &holder, tf::csg::expr e) {
  REQUIRE_THAT(sheets_law_volume(holder, e),
               Catch::Matchers::WithinAbs(0.0, 1e-9));
}

// A over [-4,4]^2 on z = 0.2 x with a hill, B the same with its middle sunk
// by 2: the pit between them holds 18
auto sheets_rim_survey(bool pit) -> sheets_mesh_t {
  auto m = tf::make_plane_mesh<sheets_index_t>(sheets_real_t(8),
                                               sheets_real_t(8), 8, 8);
  for (std::size_t i = 0; i < m.points_buffer().size(); ++i) {
    auto p = m.points_buffer()[i];
    sheets_real_t z = sheets_real_t(0.2) * p[0];
    if (std::abs(p[0] - 2) < 0.5 && std::abs(p[1] - 2) < 0.5)
      z += 2;
    if (pit && std::abs(p[0]) < 1.5 && std::abs(p[1]) < 1.5)
      z -= 2;
    m.points_buffer()[i][2] = z;
  }
  return m;
}

// the rim stands between -0.8 and 0.8, so the first box holds some of it
// exactly when its top rises above -0.8
auto sheets_rim_box_holds_rim(const std::vector<sheets_real_t> &tops) -> bool {
  return !tops.empty() && tops[0] > sheets_real_t(-0.8);
}

auto sheets_rim_pit_scene(sheets_real_t half, std::vector<sheets_real_t> tops)
    -> std::vector<sheets_mesh_t> {
  std::vector<sheets_mesh_t> meshes;
  meshes.push_back(sheets_law_box(-half, -half, -5, half, half, 5));
  meshes.push_back(sheets_rim_survey(false));
  meshes.push_back(sheets_rim_survey(true));
  for (const auto top : tops)
    meshes.push_back(sheets_law_box(-5, -5, -7, 5, 5, top));
  return meshes;
}

} // namespace

TEST_CASE("sheets: a free rim inside a volume states nothing there",
          "[csg][sheets]") {
  using tf::csg::op;
  SECTION("a floating sheet in a box: the box reads no side") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_plane(-2, -2, 2, 2, 0)}),
        {1});
    REQUIRE(holder.graph.failed().size() == 0);
    sheets_rim_empty(holder, op(0) & op(1));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       1000.0, 1e-9);
    sheets_rim_census(holder, {{1, 0}});
  }
  SECTION("a box cut edge to edge inside a box that holds the rim") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-2, -2, -2, 2, 2, 2),
                             sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_plane(-3, -3, 3, 3, 0)}),
        {2});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(2)),
                       32.0, 1e-9);
    sheets_rim_empty(holder, (op(1) - op(0)) & op(2));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(1) - op(2)),
                       968.0, 1e-9);
    sheets_rim_census(holder, {{2, 1}});
  }
  SECTION("a poked box: the rim inside, bridged by the wall it crosses") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_plane(-8, -3, 2, 3, 0)}),
        {1});
    REQUIRE(holder.graph.failed().size() == 0);
    sheets_rim_empty(holder, op(0) & op(1));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       1000.0, 1e-9);
    sheets_rim_census(holder, {{1, 0}});
  }
  SECTION("chained: the cut box inside the rim's box inside a third") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-2, -2, -2, 2, 2, 2),
                             sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_box(-8, -8, -8, 8, 8, 8),
                             sheets_law_plane(-3, -3, 3, 3, 0)}),
        {3});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(3)),
                       32.0, 1e-9);
    sheets_rim_empty(holder, (op(1) - op(0)) & op(3));
    sheets_rim_empty(holder, (op(2) - op(1)) & op(3));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(2) - op(3)),
                       4064.0, 1e-9);
    sheets_rim_census(holder, {{3, 1}, {3, 2}});
  }
  SECTION("a rim partly in the air, partly across a wall") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_plane(-3, -3, 8, 3, 0)}),
        {1});
    REQUIRE(holder.graph.failed().size() == 0);
    sheets_rim_empty(holder, op(0) & op(1));
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       1000.0, 1e-9);
    sheets_rim_census(holder, {{1, 0}});
  }
  SECTION("stacked rims: both sheets retire, a cut box keeps both sides") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-1, -1, -1, 1, 1, 1),
                             sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_plane(-2, -2, 2, 2, 0),
                             sheets_law_plane(-2, -2, 2, 2, 0)}),
        {2, 3});
    REQUIRE(holder.graph.failed().size() == 0);
    for (const int s : {2, 3}) {
      check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(s)),
                         4.0, 1e-9);
      sheets_rim_empty(holder, (op(1) - op(0)) & op(s));
    }
    sheets_rim_census(holder, {{2, 1}, {3, 1}});
  }
  SECTION("a closed sheet has no rim and keeps its sides") {
    graph_holder holder(
        sheets_law_operands({sheets_law_box(-5, -5, -5, 5, 5, 5),
                             sheets_law_box(-1, -1, -1, 1, 1, 1)}),
        {1});
    REQUIRE(holder.graph.failed().size() == 0);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) & op(1)),
                       8.0, 1e-9);
    check_closed_solid(tf::test::csg_mesh_of(holder.graph, op(0) - op(1)),
                       992.0, 1e-9);
    sheets_rim_census(holder, {});
  }
  SECTION("a pit pair whose rims lie inside the bounding box: the cut reads "
          "empty") {
    for (const auto &tops : std::vector<std::vector<sheets_real_t>>{
             {}, {-3}, {3}, {1.5}, {-1.5}, {0}, {0.5}, {-0.5}, {1.5, -1.5}}) {
      graph_holder holder(sheets_law_operands(sheets_rim_pit_scene(5, tops)),
                          {1, 2});
      REQUIRE(holder.graph.failed().size() == 0);
      sheets_rim_empty(holder, (op(0) & op(1)) - op(2));
      // a box topped at -1.5 is cut by the pit's dish, which separates it:
      // its own side of B survives inside op0
      if (std::find(tops.begin(), tops.end(), sheets_real_t(-1.5)) !=
          tops.end())
        REQUIRE_THAT(sheets_law_volume(holder, (op(0) & op(2)) - op(1)),
                     Catch::Matchers::WithinAbs(347.3720866068, 1e-9));
      else
        sheets_rim_empty(holder, (op(0) & op(2)) - op(1));
      if (sheets_rim_box_holds_rim(tops))
        sheets_rim_census(holder, {{1, 0}, {1, 3}, {2, 0}, {2, 3}});
      else
        sheets_rim_census(holder, {{1, 0}, {2, 0}});
    }
  }
  SECTION("the same pit pair inside an inset box reads the pit") {
    for (const auto &tops : std::vector<std::vector<sheets_real_t>>{
             {}, {-3}, {3}, {1.5}, {-1.5}, {0}, {0.5}, {-0.5}, {1.5, -1.5}}) {
      graph_holder holder(
          sheets_law_operands(sheets_rim_pit_scene(sheets_real_t(3.5), tops)),
          {1, 2});
      REQUIRE(holder.graph.failed().size() == 0);
      check_closed_solid(
          tf::test::csg_mesh_of(holder.graph, (op(0) & op(1)) - op(2)), 18.0,
          1e-9);
      sheets_rim_empty(holder, (op(0) & op(2)) - op(1));
      if (sheets_rim_box_holds_rim(tops))
        sheets_rim_census(holder, {{1, 3}, {2, 3}});
      else
        sheets_rim_census(holder, {});
    }
  }
}
