/**
 * @file test_csg_solids.cpp
 * @brief CSG ports of the closed-mesh boolean scenarios in
 *        tests/cut/test_boolean.cpp.
 *
 * Each scenario is run through the N-form csg_graph / make_csg_mesh
 * pipeline (build the graph once, extract each operation), and checked
 * for the same engine-agnostic invariants the boolean tests assert:
 * closed, manifold, and signed volume against the closed-form value.
 *
 * make_boolean is intentionally not referenced, so these survive the
 * eventual removal of the pairwise path. Face/point counts are not
 * checked: csg may triangulate cut regions differently from the old
 * engine, which was never a correctness property.
 *
 * boolean_op -> expression:
 *   merge            -> csg::merge(0, 1)
 *   intersection     -> csg::intersection(0, 1)
 *   left_difference  -> csg::difference(0, 1)
 *   right_difference -> csg::difference(1, 0)
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "type_traits.hpp"

#include <cmath>
#include <vector>

namespace {

constexpr double pi = tf::pi<double>;

template <typename Real> auto identity_frame() -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
}

template <typename Real> auto translation_frame(Real x, Real y, Real z)
    -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

template <typename Form> struct two_forms {
  std::vector<Form> forms;
  decltype(tf::make_csg_graph(
      tf::make_range(std::declval<std::vector<Form> &>()))) graph;

  explicit two_forms(std::vector<Form> f)
      : forms(std::move(f)),
        graph(tf::make_csg_graph(tf::make_range(forms))) {}

  two_forms(const two_forms &) = delete;
  two_forms &operator=(const two_forms &) = delete;
};

template <typename Real, typename Mesh>
auto two_form_graph(const Mesh &m0, const Mesh &m1,
                    const tf::frame<Real, 3> &f1) {
  using form_t =
      decltype(m0.polygons() | tf::tag(std::declval<tf::frame<Real, 3>>()));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(m0.polygons() | tf::tag(identity_frame<Real>()));
  forms.push_back(m1.polygons() | tf::tag(f1));
  return two_forms<form_t>(std::move(forms));
}

template <typename Graph>
void check_solid(Graph &graph, const tf::csg::expr &e, double expected,
                 double tol) {
  auto m = tf::make_csg_mesh(graph, e);
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
               Catch::Matchers::WithinAbs(expected, tol));
}

template <typename Graph> void check_empty(Graph &graph, const tf::csg::expr &e) {
  auto m = tf::make_csg_mesh(graph, e);
  REQUIRE(m.polygons().size() == 0);
}

} // namespace

// ============================================================================
// Bicylinder (Steinmetz) intersection of two crossed cylinders.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: bicylinder intersection", "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  real_t radius = real_t(1);
  real_t height = real_t(4);
  auto cyl = tf::triangulated(
      tf::make_cylinder_mesh<index_t>(radius, height, 400).polygons());
  tf::ensure_positive_orientation(cyl.polygons());

  auto center = tf::centroid(cyl.polygons());
  auto rot = tf::make_frame(
      tf::make_rotation(tf::deg(real_t(90)), tf::axis<0>, center));

  auto holder = two_form_graph<real_t>(cyl, cyl, rot);
  auto &graph = holder.graph;

  // Steinmetz solid volume = 16 r^3 / 3.
  double expected = 16.0 * std::pow(double(radius), 3) / 3.0;
  check_solid(graph, tf::csg::intersection(0, 1), expected, expected * 0.01);
}

// ============================================================================
// Nested spheres: inner fully inside outer.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: nested spheres", "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  real_t outer_r = real_t(2), inner_r = real_t(1);
  auto outer = tf::make_sphere_mesh<index_t>(outer_r, 120, 120);
  auto inner = tf::make_sphere_mesh<index_t>(inner_r, 100, 100);
  tf::ensure_positive_orientation(outer.polygons());
  tf::ensure_positive_orientation(inner.polygons());

  double vo = (4.0 / 3.0) * pi * std::pow(double(outer_r), 3);
  double vi = (4.0 / 3.0) * pi * std::pow(double(inner_r), 3);

  auto holder = two_form_graph<real_t>(outer, inner, identity_frame<real_t>());
  auto &graph = holder.graph;

  SECTION("union = outer") {
    check_solid(graph, tf::csg::merge(0, 1), vo, vo * 0.01);
  }
  SECTION("difference = hollow shell") {
    check_solid(graph, tf::csg::difference(0, 1), vo - vi, (vo - vi) * 0.01);
  }
  SECTION("intersection = inner") {
    check_solid(graph, tf::csg::intersection(0, 1), vi, vi * 0.01);
  }
}

// ============================================================================
// Overlapping unit boxes, second translated by (0.5, 0, 0). Overlap 0.5.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: overlapping boxes", "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto b0 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  auto b1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::ensure_positive_orientation(b0.polygons());
  tf::ensure_positive_orientation(b1.polygons());

  auto f1 = translation_frame<real_t>(real_t(0.5), real_t(0), real_t(0));
  auto holder = two_form_graph<real_t>(b0, b1, f1);
  auto &graph = holder.graph;

  const double overlap = 0.5, box = 1.0, tol = 0.02;
  SECTION("union") {
    check_solid(graph, tf::csg::merge(0, 1), 2 * box - overlap, tol);
  }
  SECTION("intersection") {
    check_solid(graph, tf::csg::intersection(0, 1), overlap, tol);
  }
  SECTION("left difference") {
    check_solid(graph, tf::csg::difference(0, 1), box - overlap, tol);
  }
  SECTION("right difference") {
    check_solid(graph, tf::csg::difference(1, 0), box - overlap, tol);
  }
}

// ============================================================================
// Non-overlapping unit boxes, second far away.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: non-overlapping boxes", "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto b0 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  auto b1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::ensure_positive_orientation(b0.polygons());
  tf::ensure_positive_orientation(b1.polygons());

  auto f1 = translation_frame<real_t>(real_t(5), real_t(0), real_t(0));
  auto holder = two_form_graph<real_t>(b0, b1, f1);
  auto &graph = holder.graph;

  SECTION("union = sum") {
    check_solid(graph, tf::csg::merge(0, 1), 2.0, 0.02);
  }
  SECTION("intersection = empty") {
    check_empty(graph, tf::csg::intersection(0, 1));
  }
  SECTION("left difference = box0") {
    check_solid(graph, tf::csg::difference(0, 1), 1.0, 0.02);
  }
}

// ============================================================================
// Overlapping equal spheres, centers separated by 1. Lens intersection.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: overlapping spheres", "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  real_t radius = real_t(1), sep = real_t(1);
  auto s0 = tf::make_sphere_mesh<index_t>(radius, 60, 60);
  auto s1 = tf::make_sphere_mesh<index_t>(radius, 60, 60);
  tf::ensure_positive_orientation(s0.polygons());
  tf::ensure_positive_orientation(s1.polygons());

  double sv = (4.0 / 3.0) * pi * std::pow(double(radius), 3);
  double h = 2.0 * double(radius) - double(sep);
  double lens = (pi * h * h / 12.0) * (6.0 * double(radius) - h);

  auto f1 = translation_frame<real_t>(sep, real_t(0), real_t(0));
  auto holder = two_form_graph<real_t>(s0, s1, f1);
  auto &graph = holder.graph;

  SECTION("union = 2*sphere - lens") {
    check_solid(graph, tf::csg::merge(0, 1), 2 * sv - lens, (2 * sv - lens) * 0.02);
  }
  SECTION("intersection = lens") {
    check_solid(graph, tf::csg::intersection(0, 1), lens, lens * 0.02);
  }
  SECTION("left difference = sphere - lens") {
    check_solid(graph, tf::csg::difference(0, 1), sv - lens, (sv - lens) * 0.02);
  }
}

// ============================================================================
// Multi-component operand 0 (two disjoint outer spheres), operand 1 a small
// sphere inside the left one.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: multi-component vs inner sphere",
                   "[csg][solids]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  real_t outer_r = real_t(2), inner_r = real_t(1);
  auto left = tf::make_sphere_mesh<index_t>(outer_r, 50, 50);
  auto right = tf::make_sphere_mesh<index_t>(outer_r, 50, 50);
  tf::ensure_positive_orientation(left.polygons());
  tf::ensure_positive_orientation(right.polygons());

  auto two_spheres = tf::concatenated(
      left.polygons(),
      right.polygons() |
          tf::tag(translation_frame<real_t>(real_t(10), real_t(0), real_t(0))));

  auto inner = tf::make_sphere_mesh<index_t>(inner_r, 40, 40);
  tf::ensure_positive_orientation(inner.polygons());

  double vo = (4.0 / 3.0) * pi * std::pow(double(outer_r), 3);
  double vi = (4.0 / 3.0) * pi * std::pow(double(inner_r), 3);

  auto holder =
      two_form_graph<real_t>(two_spheres, inner, identity_frame<real_t>());
  auto &graph = holder.graph;

  SECTION("union = two outer spheres") {
    check_solid(graph, tf::csg::merge(0, 1), 2 * vo, (2 * vo) * 0.02);
  }
  SECTION("left difference = (outer - inner) + outer") {
    check_solid(graph, tf::csg::difference(0, 1), (vo - vi) + vo,
                ((vo - vi) + vo) * 0.02);
  }
  SECTION("intersection = inner") {
    check_solid(graph, tf::csg::intersection(0, 1), vi, vi * 0.02);
  }
}
