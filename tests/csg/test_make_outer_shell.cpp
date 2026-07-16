/**
 * @file test_make_outer_shell.cpp
 * @brief Outer-shell repair of self-intersecting solids.
 *
 * make_outer_shell arranges a mesh at its self-intersection curves,
 * labels the volumetric domains, and keeps only the faces bounding the
 * unbounded outside, oriented outward. Scenarios: interpenetrating
 * shells (the spine-screw failure shape) checked against the pairwise
 * boolean union, clean passthrough, inward-oriented input re-oriented
 * outward, and nested-cavity removal.
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

#include <utility>

namespace {

template <typename Real> auto translation_frame(Real x, Real y, Real z)
    -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

template <typename Polygons> void check_watertight(const Polygons &polygons) {
  REQUIRE(tf::is_closed(polygons));
  REQUIRE(tf::is_manifold(polygons));
  auto curves = tf::make_self_intersection_curves(polygons);
  REQUIRE(curves.size() == 0);
}

} // namespace

TEMPLATE_TEST_CASE("make_outer_shell: interpenetrating closed shells",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto frame = translation_frame(real_t(0.8), real_t(0), real_t(0));

  auto [merged, labels, face_labels] = tf::make_boolean(
      sf.polygons(), sf.polygons() | tf::tag(frame), tf::boolean_op::merge);
  auto expected = static_cast<double>(tf::signed_volume(merged.polygons()));

  // One mesh, two interpenetrating closed shells: the shape that breaks
  // the inside/outside toggle invariant for downstream classification.
  auto soup =
      tf::concatenated(sf.polygons(), sf.polygons() | tf::tag(frame));
  auto shell = tf::make_outer_shell(soup.polygons());

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: clean solid passes through",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto expected = static_cast<double>(tf::signed_volume(sf.polygons()));

  auto shell = tf::make_outer_shell(sf.polygons());

  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == sf.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: inward-oriented input comes out outward",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto expected = static_cast<double>(tf::signed_volume(sf.polygons()));
  REQUIRE(expected > 0.0);

  auto &face_data = sf.faces_buffer().data_buffer();
  for (std::size_t i = 0; i < face_data.size(); i += 3)
    std::swap(face_data[i], face_data[i + 1]);
  REQUIRE(static_cast<double>(tf::signed_volume(sf.polygons())) < 0.0);

  auto shell = tf::make_outer_shell(sf.polygons());

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: open fragment noise is dropped",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto expected = static_cast<double>(tf::signed_volume(sf.polygons()));

  // An open patch piercing the sphere surface: junk that must not
  // survive into the shell under either open-fragment mode.
  auto patch = tf::triangulated(
      tf::make_plane_mesh<index_t>(real_t(3), real_t(3)).polygons());
  auto soup = tf::concatenated(sf.polygons(), patch.polygons());

  auto default_config =
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours};
  for (auto flags : {tf::domain_config::none,
                     tf::domain_config::ignore_open_fragments}) {
    auto shell = tf::make_outer_shell(soup.polygons(), default_config, flags);
    check_watertight(shell.polygons());
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
                 Catch::Matchers::WithinRel(expected, 1e-6));
  }
}

TEMPLATE_TEST_CASE("make_outer_shell: nested cavity is removed",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto outer = tf::make_sphere_mesh<index_t>(real_t(2), 32, 32);
  tf::ensure_positive_orientation(outer.polygons());
  auto inner = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  tf::ensure_positive_orientation(inner.polygons());
  auto expected = static_cast<double>(tf::signed_volume(outer.polygons()));

  auto soup = tf::concatenated(outer.polygons(), inner.polygons());
  auto shell = tf::make_outer_shell(soup.polygons());

  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == outer.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form csg graph matches the mesh path",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto frame = translation_frame(real_t(0.8), real_t(0), real_t(0));
  auto soup = tf::concatenated(sf.polygons(), sf.polygons() | tf::tag(frame));

  auto mesh_path = tf::make_outer_shell(soup.polygons());
  auto expected = static_cast<double>(tf::signed_volume(mesh_path.polygons()));

  // The graph IS the self arrangement: one form, no expression.
  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph, clean solid passes "
                   "through",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  tf::ensure_positive_orientation(sf.polygons());
  auto expected = static_cast<double>(tf::signed_volume(sf.polygons()));

  // single-form overload: no range wrap
  auto graph = tf::make_csg_graph(sf.polygons());
  auto shell = tf::make_outer_shell(graph);

  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == sf.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-9));
}

TEMPLATE_TEST_CASE("csg domains: one-form graph decomposes the self "
                   "arrangement",
                   "[csg][domains][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto frame = translation_frame(real_t(0.8), real_t(0), real_t(0));
  auto soup = tf::concatenated(sf.polygons(), sf.polygons() | tf::tag(frame));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto [domains, ids] = tf::make_csg_domains(graph);

  // two interpenetrating spheres: lens + two crescents
  REQUIRE(ids.size() == 3);
  double total = 0;
  for (auto &d : domains) {
    check_watertight(d.polygons());
    total += static_cast<double>(tf::signed_volume(d.polygons()));
  }
  // cells tile the enclosed union: total == outer shell volume
  auto shell = tf::make_outer_shell(graph);
  REQUIRE_THAT(total,
               Catch::Matchers::WithinRel(
                   static_cast<double>(tf::signed_volume(shell.polygons())),
                   1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph, disjoint components",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto frame = translation_frame(real_t(3), real_t(0), real_t(0));
  auto soup = tf::concatenated(sf.polygons(), sf.polygons() | tf::tag(frame));
  auto expected = 2.0 * static_cast<double>(tf::signed_volume(sf.polygons()));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  // BOTH disjoint shells must survive: the universe is one region but
  // several fine domains, related only through the nesting merges.
  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  auto [domains, ids] = tf::make_csg_domains(graph);
  REQUIRE(ids.size() == 2);
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph, nested contact-free "
                   "shells",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto outer = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(outer.polygons());
  auto inner = tf::make_sphere_mesh<index_t>(real_t(0.5), 32, 32);
  tf::ensure_positive_orientation(inner.polygons());
  auto soup = tf::concatenated(outer.polygons(), inner.polygons());
  auto expected = static_cast<double>(tf::signed_volume(outer.polygons()));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  // The inner shell is enclosed: only the outer wall bounds the union.
  // Requires the inner bundle's nesting cast against its own tag.
  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == outer.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  // cells: the gap (outer minus ball, with its cavity wall) + the ball
  auto [domains, ids] = tf::make_csg_domains(graph);
  REQUIRE(ids.size() == 2);
  double total = 0;
  for (auto &d : domains) {
    check_watertight(d.polygons());
    total += static_cast<double>(tf::signed_volume(d.polygons()));
  }
  REQUIRE_THAT(total, Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph, nested pair plus a "
                   "free-floating shell",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto outer = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(outer.polygons());
  auto inner = tf::make_sphere_mesh<index_t>(real_t(0.5), 32, 32);
  tf::ensure_positive_orientation(inner.polygons());
  auto frame = translation_frame(real_t(4), real_t(0), real_t(0));
  auto soup = tf::concatenated(
      tf::concatenated(outer.polygons(), inner.polygons()).polygons(),
      outer.polygons() | tf::tag(frame));
  auto expected =
      2.0 * static_cast<double>(tf::signed_volume(outer.polygons()));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  // One nesting merge (inner's outside -> the gap) coexists with an
  // un-merged universe root (the free shell's outside): the membership
  // union-find must relate all three exteriors to one universe class.
  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == 2 * outer.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  // cells: the gap, the ball, the free shell's interior
  auto [domains, ids] = tf::make_csg_domains(graph);
  REQUIRE(ids.size() == 3);
  double total = 0;
  for (auto &d : domains) {
    check_watertight(d.polygons());
    total += static_cast<double>(tf::signed_volume(d.polygons()));
  }
  REQUIRE_THAT(total, Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: self-intersecting bundle nested in a "
                   "shell",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // Two interpenetrating spheres are ONE bundle (their pieces unite
  // across the intersection curve), and every face near the curve is
  // cut. That bundle's nesting cast toward the enclosing shell crosses
  // its own cut faces — which must not toggle the parity it reads.
  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto frame = translation_frame(real_t(0.8), real_t(0), real_t(0));
  auto blob = tf::concatenated(sf.polygons(), sf.polygons() | tf::tag(frame));
  auto big = tf::make_sphere_mesh<index_t>(real_t(4), 32, 32);
  tf::ensure_positive_orientation(big.polygons());
  auto soup = tf::concatenated(blob.polygons(), big.polygons());
  auto expected = static_cast<double>(tf::signed_volume(big.polygons()));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  // Only the enclosing wall bounds the union; the blob is interior.
  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == big.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  // cells: the gap around the blob, two crescents, the lens. The
  // crescents' crease at the intersection circle may self-touch after
  // deconversion (the global lattice is scaled by the enclosing shell),
  // so interior cells check closed + manifold + tiling, not residual
  // self-intersections.
  auto [domains, ids] = tf::make_csg_domains(graph);
  REQUIRE(ids.size() == 4);
  double total = 0;
  for (auto &d : domains) {
    REQUIRE(tf::is_closed(d.polygons()));
    REQUIRE(tf::is_manifold(d.polygons()));
    total += static_cast<double>(tf::signed_volume(d.polygons()));
  }
  REQUIRE_THAT(total, Catch::Matchers::WithinRel(expected, 1e-6));
}

TEMPLATE_TEST_CASE("make_outer_shell: twin vertex with degenerate sliver "
                   "stays sealed",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // a coordinate-twin vertex (same position, distinct id) plus a
  // zero-length-edge sliver referencing both ids: the vertex
  // identification collapses the sliver's base loop, and its
  // descriptor must land in the cut set — re-emitting the original
  // splits the star into a seam
  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  tf::ensure_positive_orientation(sf.polygons());
  auto twin_src = sf.points_buffer()[10];
  const index_t twin = index_t(sf.points_buffer().size());
  sf.points_buffer().push_back(twin_src);
  // rewire part of vertex 10's fan to the twin id
  auto &fdata = sf.faces_buffer().data_buffer();
  index_t rewired = 0;
  for (std::size_t k = 0; k < std::size_t(fdata.size()) && rewired < 3; ++k)
    if (fdata[k] == index_t(10)) {
      fdata[k] = twin;
      ++rewired;
    }
  // the degenerate sliver joining the twins
  auto nb = fdata[1];
  sf.faces_buffer().emplace_back(index_t(10), twin, nb);

  auto frame = translation_frame(real_t(0.8), real_t(0), real_t(0));
  auto other = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  tf::ensure_positive_orientation(other.polygons());
  auto soup =
      tf::concatenated(sf.polygons(), other.polygons() | tf::tag(frame));

  auto graph = tf::make_csg_graph(soup.polygons());
  auto shell = tf::make_outer_shell(graph);
  REQUIRE(tf::is_closed(shell.polygons()));
  REQUIRE(tf::is_manifold(shell.polygons()));
  auto curves = tf::make_self_intersection_curves(shell.polygons());
  REQUIRE(curves.size() == 0);
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph drops open-fragment "
                   "noise structurally",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());

  // floating open patch inside the sphere: the arrangement self-merges
  // an open component's two sides, so its wall never separates domains
  // and vanishes from the shell without any config
  tf::polygons_buffer<index_t, real_t, 3, 3> patch;
  patch.points_buffer().allocate(3);
  patch.points_buffer()[0] = tf::point<real_t, 3>{0.1f, 0.0f, 0.0f};
  patch.points_buffer()[1] = tf::point<real_t, 3>{0.0f, 0.1f, 0.0f};
  patch.points_buffer()[2] = tf::point<real_t, 3>{0.0f, 0.0f, 0.1f};
  patch.faces_buffer().emplace_back(0, 1, 2);
  auto soup = tf::concatenated(sf.polygons(), patch.polygons());

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == sf.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(
                   static_cast<double>(tf::signed_volume(sf.polygons())),
                   1e-9));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph normalizes inward "
                   "orientation",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto expected = static_cast<double>(tf::signed_volume(sf.polygons()));

  auto inward = sf;
  for (index_t f = 0; f < index_t(inward.faces_buffer().size()); ++f) {
    auto face = inward.faces_buffer()[f];
    std::swap(face[0], face[2]);
  }
  REQUIRE(static_cast<double>(tf::signed_volume(inward.polygons())) < 0);

  std::array<decltype(inward.polygons()), 1> forms = {inward.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  // volumes flip sign with the winding AND the side, so the universe
  // argmin is orientation-free and the chosen-side emit normalizes
  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-9));
}

TEMPLATE_TEST_CASE("make_outer_shell: one-form graph, mixed orientations "
                   "across disjoint components",
                   "[csg][outer_shell][graph]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());
  auto inward = sf;
  for (index_t f = 0; f < index_t(inward.faces_buffer().size()); ++f) {
    auto face = inward.faces_buffer()[f];
    std::swap(face[0], face[2]);
  }
  auto frame = translation_frame(real_t(3), real_t(0), real_t(0));
  auto soup =
      tf::concatenated(sf.polygons(), inward.polygons() | tf::tag(frame));
  auto expected = 2.0 * static_cast<double>(tf::signed_volume(sf.polygons()));

  std::array<decltype(soup.polygons()), 1> forms = {soup.polygons()};
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto shell = tf::make_outer_shell(graph);

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}
