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
// A floater behind the sheet that nothing encloses joins the outside: its
// envelope must not survive enumeration as an inverted shell.
// ============================================================================
TEST_CASE("sheets: un-enclosed behind-side floater leaves no pseudo-domain",
          "[csg][sheets]") {
  auto straddle = tf::make_sphere_mesh<Index>(Real(1), 32, 32);
  auto below = tf::make_sphere_mesh<Index>(Real(0.5), 32, 32);
  tf::ensure_positive_orientation(straddle.polygons());
  tf::ensure_positive_orientation(below.polygons());
  auto plane = tf::make_plane_mesh<Index>(Real(4), Real(4));

  std::vector<form_t> forms;
  forms.push_back(straddle.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(below.polygons() | tf::tag(frame_at(0, 0, -2)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {2});

  auto [cells, ids] = tf::make_csg_domains(holder.graph);
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

// ============================================================================
// Without a sheet declaration an open plane is inert: volume semantics
// fuse open fragments away, so it neither cuts nor contains.
// ============================================================================
TEST_CASE("sheets: undeclared open plane is inert", "[csg][sheets]") {
  auto sphere = tf::make_sphere_mesh<Index>(Real(1), 48, 48);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));
  const double full = double(tf::signed_volume(sphere.polygons()));

  std::vector<form_t> forms;
  forms.push_back(sphere.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  check_closed_solid(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)),
                     full, 1e-9);
  REQUIRE(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)).size() == 0);
}

namespace {

auto area_of(const mesh_t &m) -> double {
  double a = 0;
  auto polys = m.polygons();
  for (std::size_t f = 0; f < polys.size(); ++f) {
    auto poly = polys[f];
    tf::vector<Real, 3> e0 = poly[1] - poly[0];
    tf::vector<Real, 3> e1 = poly[2] - poly[0];
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
  auto sphere = tf::make_sphere_mesh<Index>(Real(1), 96, 96);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));
  const double full = double(tf::signed_volume(sphere.polygons()));

  std::vector<form_t> forms;
  forms.push_back(sphere.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {1});
  auto &graph = holder.graph;

  SECTION("plane & sphere = lower half-ball: hemisphere + cap disc") {
    auto m = tf::make_csg_mesh(graph, tf::csg::intersection(1, 0));
    check_closed_solid(m, full / 2, 0.01 * full);
    REQUIRE(centroid_z(m) < 0.0);
    REQUIRE_THAT(area_of(m),
                 Catch::Matchers::WithinAbs(3 * pi, 0.01 * 3 * pi));
  }
  SECTION("plane - sphere = boundary of the unbounded region below") {
    auto m = tf::make_csg_mesh(graph, tf::csg::difference(1, 0));
    REQUIRE(m.size() > 0);
    REQUIRE_FALSE(tf::is_closed(m.polygons()));
    // punctured plane (36 - pi) + hemisphere wall (2 pi)
    REQUIRE_THAT(area_of(m),
                 Catch::Matchers::WithinAbs(36 + pi, 0.01 * (36 + pi)));
  }
}

// ============================================================================
// A composite expression over disconnected volumes: the sheet halves the
// straddling sphere while whole floaters keep or lose membership by
// side, each output piece closed on its own.
// ============================================================================
TEST_CASE("sheets: composite cut of disconnected volumes", "[csg][sheets]") {
  auto straddle = tf::make_sphere_mesh<Index>(Real(1), 48, 48);
  auto above = tf::make_sphere_mesh<Index>(Real(0.5), 48, 48);
  auto below = tf::make_sphere_mesh<Index>(Real(0.5), 48, 48);
  for (auto *s : {&straddle, &above, &below})
    tf::ensure_positive_orientation(s->polygons());
  auto plane = tf::make_plane_mesh<Index>(Real(6), Real(6));
  const double full = double(tf::signed_volume(straddle.polygons()));
  const double small = double(tf::signed_volume(above.polygons()));

  std::vector<form_t> forms;
  forms.push_back(straddle.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(above.polygons() | tf::tag(frame_at(0, 0, 2)));
  forms.push_back(below.polygons() | tf::tag(frame_at(0, 0, -2)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {3});
  auto &graph = holder.graph;

  auto solids = tf::csg::merge(tf::csg::merge(0, 1), 2);
  auto check_two_pieces = [&](const mesh_t &m, double sign) {
    REQUIRE_THAT(volume_of(m), Catch::Matchers::WithinAbs(
                                   full / 2 + small, 0.01 * full));
    auto [labels, n] = tf::make_manifold_edge_connected_component_labels(
        m.polygons());
    REQUIRE(n == Index(2));
    auto [pieces, ids] = tf::split_into_components(m.polygons(), labels);
    for (auto &piece : pieces) {
      REQUIRE(tf::is_closed(piece.polygons()));
      REQUIRE(tf::is_manifold(piece.polygons()));
      REQUIRE(sign * centroid_z(piece) > 0.0);
      const double v = volume_of(piece);
      const bool is_half = std::abs(v - full / 2) < 0.01 * full;
      const bool is_floater = std::abs(v - small) < 0.01 * small;
      REQUIRE((is_half || is_floater));
    }
  };

  SECTION("difference keeps upper half + above floater") {
    check_two_pieces(
        tf::make_csg_mesh(graph, tf::csg::difference(solids, 3)), +1.0);
  }
  SECTION("intersection keeps lower half + below floater") {
    check_two_pieces(
        tf::make_csg_mesh(graph, tf::csg::intersection(solids, 3)), -1.0);
  }
}

// ============================================================================
// Coincident duplicate sheets: the outside must stay excluded.
// ============================================================================

namespace {

auto reversed_copy(const mesh_t &m) -> mesh_t {
  mesh_t r = m;
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
  auto boxq = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)); // vol 8
  mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<Index>(Real(4), Real(4)); // z=0, +z
  mesh_t plane = tf::triangulated(planeq.polygons());
  mesh_t plane_rev = reversed_copy(plane);

  auto check_halves = [&](mesh_t &second_sheet) {
    std::vector<form_t> forms;
    forms.push_back(box.polygons() | tf::tag(frame_at(0, 0, 0)));
    forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
    forms.push_back(second_sheet.polygons() | tf::tag(frame_at(0, 0, 0)));
    graph_holder<2> holder(std::move(forms), {1, 2});
    auto [cells, ids] = tf::make_csg_domains(holder.graph);
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
  auto boxq = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<Index>(Real(4), Real(4)); // z=0, +z
  mesh_t plane = tf::triangulated(planeq.polygons());
  mesh_t plane_rev = reversed_copy(plane);

  std::vector<form_t> forms;
  forms.push_back(box.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane_rev.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<2> holder(std::move(forms), {1, 2});

  auto centroid_of_selection = [&](const tf::csg::expr &e) {
    auto [cells, ids] = tf::make_csg_domains(holder.graph, e);
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
  auto boxq = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<Index>(Real(4), Real(4)); // z=0, +z
  mesh_t plane = tf::triangulated(planeq.polygons());
  mesh_t plane_rev = reversed_copy(plane);

  std::vector<form_t> forms;
  forms.push_back(box.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane_rev.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<3> holder(std::move(forms), {1, 2, 3});

  auto [cells, ids] = tf::make_csg_domains(holder.graph);
  REQUIRE(cells.size() == 2);

  auto half_behind = [&](int op_id) {
    auto [sel, sel_ids] =
        tf::make_csg_domains(holder.graph, tf::csg::op(0) & tf::csg::op(op_id));
    REQUIRE(sel.size() == 1);
    return centroid_z(sel[0]);
  };
  REQUIRE(half_behind(1) < 0.0); // behind +z normal = below
  REQUIRE(half_behind(2) > 0.0); // reversed copy: behind = above
  REQUIRE(half_behind(3) < 0.0); // forward copy: below again
}

TEST_CASE("sheets: reversed single sheet anchors behind its own normal",
          "[csg][sheets]") {
  auto boxq = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  mesh_t box = tf::triangulated(boxq.polygons());
  auto planeq = tf::make_plane_mesh<Index>(Real(4), Real(4)); // z=0, +z
  mesh_t plane_rev = reversed_copy(tf::triangulated(planeq.polygons()));

  std::vector<form_t> forms;
  forms.push_back(box.polygons() | tf::tag(frame_at(0, 0, 0)));
  forms.push_back(plane_rev.polygons() | tf::tag(frame_at(0, 0, 0)));
  graph_holder<1> holder(std::move(forms), {1});

  auto [cells, ids] = tf::make_csg_domains(holder.graph);
  REQUIRE(cells.size() == 2);

  // Normal points -z, so behind the sheet is the upper half.
  auto [sel, sel_ids] =
      tf::make_csg_domains(holder.graph, tf::csg::op(0) & tf::csg::op(1));
  REQUIRE(sel.size() == 1);
  REQUIRE_THAT(std::abs(double(tf::signed_volume(sel[0].polygons()))),
               Catch::Matchers::WithinAbs(4.0, 1e-9));
  REQUIRE(centroid_z(sel[0]) > 0.0);
}
