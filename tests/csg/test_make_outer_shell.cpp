/**
 * @file test_make_outer_shell.cpp
 * @brief Outer-shell repair of self-intersecting solids.
 *
 * make_outer_shell arranges a mesh at its self-intersection curves,
 * labels the volumetric domains, and keeps only the faces bounding the
 * unbounded outside, oriented outward. Scenarios: interpenetrating
 * shells (the spine-screw failure shape) checked against the pairwise
 * boolean union, clean passthrough, inward-oriented input re-oriented
 * outward, nested-cavity removal, and integer-coordinate output.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "type_traits.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace {

template <typename Real> auto translation_frame(Real x, Real y, Real z)
    -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{x, y, z}));
}

template <typename Polygons> void check_watertight(const Polygons &polygons) {
  REQUIRE(tf::is_closed(polygons));
  REQUIRE(tf::is_manifold(polygons));
  // crossing oracle: primitives on a deconverted (float-output) mesh states
  // the seam's twin contacts as curves — an output-rounding property of the
  // float output, not a self-crossing
  auto curves = tf::make_self_intersection_curves(
      polygons, tf::intersect_config{tf::intersect_mode::sos |
                                     tf::intersect_mode::resolve_contours});
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

TEMPLATE_TEST_CASE("make_outer_shell: an uncut vertex keeps its input "
                   "coordinate",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // The shell is read off the csg graph, which materialises only created
  // points from the lattice; an original vertex reaches the output
  // untouched. Running the whole result through the lattice and
  // deconverting it back would move 174 of this sphere's 994 vertices by
  // one ulp.
  auto sf = tf::make_sphere_mesh<index_t>(real_t(1), 32, 32);
  tf::ensure_positive_orientation(sf.polygons());

  auto shell = tf::make_outer_shell(sf.polygons());

  // A clean solid is uncut: the shell's point set IS the input's, up to
  // the order the output stream discovered it.
  REQUIRE(shell.points_buffer().size() == sf.points_buffer().size());
  auto collect = [](const auto &buffer) {
    std::vector<std::array<real_t, 3>> out;
    out.reserve(buffer.size());
    for (auto p : buffer)
      out.push_back({real_t(p[0]), real_t(p[1]), real_t(p[2])});
    std::sort(out.begin(), out.end());
    return out;
  };
  REQUIRE(collect(shell.points_buffer()) == collect(sf.points_buffer()));
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

  // An open patch piercing the sphere surface: junk the arrangement's
  // self-merge makes non-separating, so it never reaches the shell.
  auto patch = tf::triangulated(
      tf::make_plane_mesh<index_t>(real_t(3), real_t(3)).polygons());
  auto soup = tf::concatenated(sf.polygons(), patch.polygons());

  auto shell = tf::make_outer_shell(soup.polygons());

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
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

TEMPLATE_TEST_CASE("make_outer_shell: integer output stays on the lattice",
                   "[csg][outer_shell]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  // the lattice the arrangement runs on for this input scalar
  using lattice_t = tf::exact::resolve_int_type<tf::none_t, real_t>;

  auto box = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::ensure_positive_orientation(box.polygons());
  auto frame = translation_frame(real_t(0.5), real_t(0.5), real_t(0.5));
  auto soup = tf::concatenated(box.polygons(), box.polygons() | tf::tag(frame));

  auto shell = tf::make_outer_shell(soup.polygons());
  check_watertight(shell.polygons());
  auto expected = static_cast<double>(tf::signed_volume(shell.polygons()));

  auto [int_shell, converter] =
      tf::make_outer_shell<tf::none_t, lattice_t>(soup.polygons());

  REQUIRE(int_shell.polygons().size() > 0);
  REQUIRE(tf::is_closed(int_shell.polygons()));
  REQUIRE(tf::is_manifold(int_shell.polygons()));
  REQUIRE(int_shell.polygons().size() == shell.polygons().size());
  REQUIRE(int_shell.points_buffer().size() == shell.points_buffer().size());

  // the returned converter is the one that produced the lattice, so
  // deconverting the integer shell reproduces the float overload's mesh
  tf::polygons_buffer<index_t, real_t, 3, 3> back;
  for (auto p : int_shell.points_buffer())
    back.points_buffer().push_back(
        converter.deconvert(tf::point<lattice_t, 3>{p[0], p[1], p[2]}));
  for (auto face : int_shell.faces_buffer())
    back.faces_buffer().emplace_back(face[0], face[1], face[2]);

  check_watertight(back.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(back.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-9));
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
  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

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
  auto sf_operand = tf::test::make_tagged_operand(sf);
  auto graph = tf::test::build_self_csg_graph(sf_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto [domains, ids] = tf::test::csg_domains_of(graph);

  // two interpenetrating spheres: lens + two crescents
  REQUIRE(ids.size() == 3);
  double total = 0;
  for (auto &d : domains) {
    check_watertight(d.polygons());
    total += static_cast<double>(tf::signed_volume(d.polygons()));
  }
  // cells tile the enclosed union: total == outer shell volume
  auto shell = tf::test::outer_shell_of(graph);
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

  // BOTH disjoint shells must survive: the universe is one region but
  // several fine domains, related only through the nesting merges.
  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  auto [domains, ids] = tf::test::csg_domains_of(graph);
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

  // The inner shell is enclosed: only the outer wall bounds the union.
  // Requires the inner bundle's nesting cast against its own tag.
  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == outer.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  // cells: the gap (outer minus ball, with its cavity wall) + the ball
  auto [domains, ids] = tf::test::csg_domains_of(graph);
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

  // One nesting merge (inner's outside -> the gap) coexists with an
  // un-merged universe root (the free shell's outside): the membership
  // union-find must relate all three exteriors to one universe class.
  check_watertight(shell.polygons());
  REQUIRE(shell.polygons().size() == 2 * outer.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));

  // cells: the gap, the ball, the free shell's interior
  auto [domains, ids] = tf::test::csg_domains_of(graph);
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

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
  auto [domains, ids] = tf::test::csg_domains_of(graph);
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
  // materialize: [] yields a view, and push_back may reallocate under it
  const tf::point<real_t, 3> twin_src{sf.points_buffer()[10][0],
                                      sf.points_buffer()[10][1],
                                      sf.points_buffer()[10][2]};
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);
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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

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

  auto inward_operand = tf::test::make_tagged_operand(inward);
  auto graph = tf::test::build_self_csg_graph(inward_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

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

  auto soup_operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(soup_operand.form(), {});
  auto shell = tf::test::outer_shell_of(graph);

  check_watertight(shell.polygons());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(shell.polygons())),
               Catch::Matchers::WithinRel(expected, 1e-6));
}
