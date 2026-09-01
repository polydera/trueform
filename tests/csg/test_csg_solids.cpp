/**
 * @file test_csg_solids.cpp
 * @brief The closed-mesh boolean scenarios, answered by the csg_graph
 *        engine.
 *
 * Each scenario is run through the N-form csg_graph / make_csg_mesh
 * pipeline (build the graph once, extract each operation), and checked
 * for engine-agnostic invariants: closed, manifold, and signed volume
 * against the closed-form value.
 *
 * make_boolean is not referenced here; tests/csg/test_boolean.cpp owns the
 * pairwise wrapper. Face and point counts are asserted only where they are
 * an identity — where nothing intersects, no triangulation choice can move
 * them. Wherever a cut region is triangulated the count is an engine detail,
 * never a correctness property.
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

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include "type_traits.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace {

constexpr double solids_pi = tf::pi<double>;

template <typename Real> auto identity_frame() -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
}

template <typename Real> auto translation_frame(Real x, Real y, Real z)
    -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

template <typename Operand> struct two_forms {
  using form_type = decltype(std::declval<Operand &>().form());

  std::vector<Operand> operands;
  std::vector<form_type> forms;
  decltype(tf::test::build_range_csg_graph(
      tf::test::forms_range(std::declval<std::vector<form_type> &>()),
      tf::test::no_sheets(), tf::arrangement_config{})) graph;

  two_forms(std::vector<Operand> ops, tf::arrangement_config config)
      : operands(std::move(ops)), forms(tf::test::tagged_forms(operands)),
        graph(tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                              tf::test::no_sheets(), config)) {}

  explicit two_forms(std::vector<Operand> ops)
      : two_forms(std::move(ops), tf::arrangement_config{}) {}

  two_forms(const two_forms &) = delete;
  two_forms &operator=(const two_forms &) = delete;
};

template <typename Real, typename Mesh>
auto two_form_graph(const Mesh &m0, const Mesh &m1,
                    const tf::frame<Real, 3> &f1,
                    tf::arrangement_config config = {}) {
  using operand_t = decltype(tf::test::make_tagged_operand(m0));
  std::vector<operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(m0));
  operands.push_back(tf::test::make_tagged_operand(
      m1, tf::transformation<Real, 3>(f1.transformation())));
  return two_forms<operand_t>(std::move(operands), config);
}

template <typename Graph>
auto solids_check_solid(Graph &graph, const tf::csg::expr &e, double expected,
                        double tol) {
  auto m = tf::test::csg_mesh_of(graph, e);
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
               Catch::Matchers::WithinAbs(expected, tol));
  return m;
}

template <typename Graph> void check_empty(Graph &graph, const tf::csg::expr &e) {
  auto m = tf::test::csg_mesh_of(graph, e);
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
  solids_check_solid(graph, tf::csg::intersection(0, 1), expected,
                     expected * 0.01);
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

  double vo = (4.0 / 3.0) * solids_pi * std::pow(double(outer_r), 3);
  double vi = (4.0 / 3.0) * solids_pi * std::pow(double(inner_r), 3);

  auto holder = two_form_graph<real_t>(outer, inner, identity_frame<real_t>());
  auto &graph = holder.graph;

  SECTION("union = outer") {
    auto m = solids_check_solid(graph, tf::csg::merge(0, 1), vo, vo * 0.01);
    // Nothing intersects, so nothing is cut: the union re-emits the outer
    // operand's faces and drops the enclosed inner operand whole.
    REQUIRE(m.polygons().size() == outer.polygons().size());
  }
  SECTION("difference = hollow shell") {
    auto m = solids_check_solid(graph, tf::csg::difference(0, 1), vo - vi,
                                (vo - vi) * 0.01);
    // Same reason, both shells kept: the outer operand's faces plus the
    // inner operand's, reversed into the cavity.
    REQUIRE(m.polygons().size() ==
            outer.polygons().size() + inner.polygons().size());
  }
  SECTION("intersection = inner") {
    solids_check_solid(graph, tf::csg::intersection(0, 1), vi, vi * 0.01);
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
    solids_check_solid(graph, tf::csg::merge(0, 1), 2 * box - overlap, tol);
  }
  SECTION("intersection") {
    solids_check_solid(graph, tf::csg::intersection(0, 1), overlap, tol);
  }
  SECTION("left difference") {
    solids_check_solid(graph, tf::csg::difference(0, 1), box - overlap, tol);
  }
  SECTION("right difference") {
    solids_check_solid(graph, tf::csg::difference(1, 0), box - overlap, tol);
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
    solids_check_solid(graph, tf::csg::merge(0, 1), 2.0, 0.02);
  }
  SECTION("intersection = empty") {
    check_empty(graph, tf::csg::intersection(0, 1));
  }
  SECTION("left difference = box0") {
    solids_check_solid(graph, tf::csg::difference(0, 1), 1.0, 0.02);
  }
}

// ============================================================================
// Stacked unit boxes separated by a sub-tolerance gap. The band pairs the
// facing walls at the narrow phase, the corners weld, and the union is one
// solid; a gap beyond the band leaves two intact solids. The failure this
// pins: exact plane signs let the separating reject discard near-coincident
// parallel faces while their exactly-coplanar neighbours still welded,
// yielding a partial weld — a closed but non-manifold union in four
// components.
// ============================================================================
TEMPLATE_TEST_CASE("csg_solids: stacked boxes weld across the band",
                   "[csg][solids][tolerance]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto b0 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  auto b1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::ensure_positive_orientation(b0.polygons());
  tf::ensure_positive_orientation(b1.polygons());

  const double tolerance = 1e-3;
  const auto mode = tf::intersect_mode::primitives |
                    tf::intersect_mode::resolve_contours |
                    tf::intersect_mode::within;
  auto stacked = [&](real_t gap) {
    return two_form_graph<real_t>(
        b0, b1,
        translation_frame<real_t>(real_t(0), real_t(0), real_t(1) + gap),
        tf::intersect_config{mode, tolerance});
  };

  SECTION("gap inside the band unions to one solid") {
    auto holder = stacked(real_t(0.5e-3));
    auto m = tf::test::csg_mesh_of(holder.graph, tf::csg::merge(0, 1));
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    // The welded corners collapse sixteen points to twelve.
    REQUIRE(m.points().size() == 12u);
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 Catch::Matchers::WithinAbs(2.0, 0.01));
  }

  SECTION("gap beyond the band stays two intact solids") {
    auto holder = stacked(real_t(2e-3));
    auto m = tf::test::csg_mesh_of(holder.graph, tf::csg::merge(0, 1));
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    // The separating reject still prunes beyond the band: nothing welds.
    REQUIRE(m.points().size() == 16u);
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 Catch::Matchers::WithinAbs(2.0, 0.01));
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

  double sv = (4.0 / 3.0) * solids_pi * std::pow(double(radius), 3);
  double h = 2.0 * double(radius) - double(sep);
  double lens = (solids_pi * h * h / 12.0) * (6.0 * double(radius) - h);

  auto f1 = translation_frame<real_t>(sep, real_t(0), real_t(0));
  auto holder = two_form_graph<real_t>(s0, s1, f1);
  auto &graph = holder.graph;

  SECTION("union = 2*sphere - lens") {
    solids_check_solid(graph, tf::csg::merge(0, 1), 2 * sv - lens,
                       (2 * sv - lens) * 0.02);
  }
  SECTION("intersection = lens") {
    solids_check_solid(graph, tf::csg::intersection(0, 1), lens, lens * 0.02);
  }
  SECTION("left difference = sphere - lens") {
    solids_check_solid(graph, tf::csg::difference(0, 1), sv - lens,
                       (sv - lens) * 0.02);
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

  double vo = (4.0 / 3.0) * solids_pi * std::pow(double(outer_r), 3);
  double vi = (4.0 / 3.0) * solids_pi * std::pow(double(inner_r), 3);

  auto holder =
      two_form_graph<real_t>(two_spheres, inner, identity_frame<real_t>());
  auto &graph = holder.graph;

  SECTION("union = two outer spheres") {
    solids_check_solid(graph, tf::csg::merge(0, 1), 2 * vo, (2 * vo) * 0.02);
  }
  SECTION("left difference = (outer - inner) + outer") {
    solids_check_solid(graph, tf::csg::difference(0, 1), (vo - vi) + vo,
                       ((vo - vi) + vo) * 0.02);
  }
  SECTION("intersection = inner") {
    solids_check_solid(graph, tf::csg::intersection(0, 1), vi, vi * 0.02);
  }
}
