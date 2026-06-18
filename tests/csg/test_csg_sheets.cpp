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
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace {

using Index = int;
using Real = double;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

constexpr double pi = tf::pi<double>;

auto frame_at(Real x, Real y, Real z) {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

using frame_t = decltype(frame_at(0, 0, 0));
using form_t = decltype(std::declval<mesh_t &>().polygons() |
                        tf::tag(std::declval<frame_t>()));

template <std::size_t NSheets> struct graph_holder {
  std::vector<form_t> forms;
  std::array<int, NSheets> sheets;
  decltype(tf::make_csg_graph(
      tf::make_range(std::declval<std::vector<form_t> &>()),
      tf::make_range(std::declval<std::array<int, NSheets> &>()))) graph;

  graph_holder(std::vector<form_t> f, std::array<int, NSheets> s)
      : forms(std::move(f)), sheets(s),
        graph(tf::make_csg_graph(tf::make_range(forms),
                                 tf::make_range(sheets))) {}

  graph_holder(const graph_holder &) = delete;
  graph_holder &operator=(const graph_holder &) = delete;
};

auto volume_of(const mesh_t &m) -> double {
  return m.size() ? double(tf::signed_volume(m.polygons())) : 0.0;
}

auto centroid_z(const mesh_t &m) -> double {
  double z = 0;
  std::size_t n = 0;
  for (auto p : m.points()) {
    z += double(p[2]);
    ++n;
  }
  return n ? z / double(n) : 0.0;
}

void check_closed_solid(const mesh_t &m, double expected_volume, double tol) {
  REQUIRE(m.size() > 0);
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
  REQUIRE_THAT(volume_of(m), Catch::Matchers::WithinAbs(expected_volume, tol));
}

} // namespace

// ============================================================================
// A sheet plane cuts a box into closed capped halves.
// ============================================================================
TEST_CASE("sheets: plane cuts a box into closed halves", "[csg][sheets]") {
  auto box = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)); // vol 8
  auto plane = tf::make_plane_mesh<Index>(Real(4), Real(4));      // z=0, +z

  std::vector<form_t> forms;
  forms.push_back(box.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {1});
  auto &graph = holder.graph;

  SECTION("difference keeps the half above the normal") {
    auto m = tf::make_csg_mesh(graph, tf::csg::difference(0, 1));
    check_closed_solid(m, 4.0, 1e-9);
    REQUIRE(centroid_z(m) > 0.0);
  }
  SECTION("intersection keeps the half below the normal") {
    auto m = tf::make_csg_mesh(graph, tf::csg::intersection(0, 1));
    check_closed_solid(m, 4.0, 1e-9);
    REQUIRE(centroid_z(m) < 0.0);
  }
  SECTION("union and sheet-minus-volume are honest open boundaries") {
    auto u = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
    REQUIRE(u.size() > 0);
    REQUIRE_FALSE(tf::is_closed(u.polygons()));
    auto r = tf::make_csg_mesh(graph, tf::csg::difference(1, 0));
    REQUIRE(r.size() > 0);
    REQUIRE_FALSE(tf::is_closed(r.polygons()));
  }
}

// ============================================================================
// A sheet plane cuts a sphere: the original ask.
// ============================================================================
TEST_CASE("sheets: plane cuts a sphere into closed halves", "[csg][sheets]") {
  auto sphere = tf::make_sphere_mesh<Index>(Real(1), 96, 96);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane = tf::make_plane_mesh<Index>(Real(3), Real(3));

  std::vector<form_t> forms;
  forms.push_back(sphere.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {1});
  auto &graph = holder.graph;

  const double half = (4.0 / 3.0) * pi / 2.0;
  auto upper = tf::make_csg_mesh(graph, tf::csg::difference(0, 1));
  auto lower = tf::make_csg_mesh(graph, tf::csg::intersection(0, 1));
  check_closed_solid(upper, half, half * 0.01);
  check_closed_solid(lower, half, half * 0.01);
  REQUIRE(centroid_z(upper) > 0.0);
  REQUIRE(centroid_z(lower) < 0.0);
}

// ============================================================================
// A clean closed mesh declared as a sheet is a no-op.
// ============================================================================
TEST_CASE("sheets: closed mesh as sheet is a no-op", "[csg][sheets]") {
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));

  auto make_forms = [&] {
    std::vector<form_t> forms;
    forms.push_back(a.polygons() | tf::tag(frame_at(0, 0, 0)));
    forms.push_back(b.polygons() | tf::tag(frame_at(1, 0, 0)));
    return forms;
  };

  auto volumes = std::vector<std::pair<tf::csg::expr, double>>{
      {tf::csg::merge(0, 1), 12.0},
      {tf::csg::intersection(0, 1), 4.0},
      {tf::csg::difference(0, 1), 4.0},
      {tf::csg::difference(1, 0), 4.0},
  };

  graph_holder<1> as_sheet(make_forms(), {1});
  auto plain_forms = make_forms();
  auto plain = tf::make_csg_graph(tf::make_range(plain_forms));

  for (auto &[e, expected] : volumes) {
    auto m_sheet = tf::make_csg_mesh(as_sheet.graph, e);
    auto m_plain = tf::make_csg_mesh(plain, e);
    check_closed_solid(m_sheet, expected, 1e-9);
    REQUIRE(m_sheet.size() == m_plain.size());
    REQUIRE_THAT(volume_of(m_sheet),
                 Catch::Matchers::WithinAbs(volume_of(m_plain), 1e-12));
  }
}

// ============================================================================
// Disconnected volumes classify by side of the sheet.
// ============================================================================
TEST_CASE("sheets: floating volumes know their side", "[csg][sheets]") {
  auto above = tf::make_box_mesh<Index>(Real(1), Real(1), Real(1)); // vol 1
  auto below = tf::make_box_mesh<Index>(Real(1), Real(1), Real(1));
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));

  std::vector<form_t> forms;
  forms.push_back(above.polygons() | tf::tag(frame_at(0, 0, 2)));
  forms.push_back(below.polygons() | tf::tag(frame_at(0, 0, -2)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {2});
  auto &graph = holder.graph;

  SECTION("above box is entirely above") {
    check_closed_solid(
        tf::make_csg_mesh(graph, tf::csg::difference(0, 2)), 1.0, 1e-9);
    REQUIRE(tf::make_csg_mesh(graph, tf::csg::intersection(0, 2)).size() == 0);
  }
  SECTION("below box is entirely below") {
    check_closed_solid(
        tf::make_csg_mesh(graph, tf::csg::intersection(1, 2)), 1.0, 1e-9);
    REQUIRE(tf::make_csg_mesh(graph, tf::csg::difference(1, 2)).size() == 0);
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
  auto box = tf::make_box_mesh<Index>(Real(1), Real(1), Real(1));
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));

  std::vector<form_t> forms;
  forms.push_back(box.polygons() | tf::tag(frame_at(2.6, -2.6, -2.5)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {1});
  auto &graph = holder.graph;

  check_closed_solid(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)),
                     1.0, 1e-9);
  REQUIRE(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)).size() == 0);
}

// ============================================================================
// Sheets compose with volume expressions.
// ============================================================================
TEST_CASE("sheets: cut of an intersection", "[csg][sheets]") {
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));

  std::vector<form_t> forms;
  forms.push_back(a.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(b.polygons() | tf::tag(frame_at(1, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {2});
  auto &graph = holder.graph;

  // (a ^ b) has volume 4; the sheet at z=0 halves it.
  auto upper = tf::make_csg_mesh(
      graph, tf::csg::difference(tf::csg::intersection(0, 1), 2));
  check_closed_solid(upper, 2.0, 1e-9);
  REQUIRE(centroid_z(upper) > 0.0);
}
