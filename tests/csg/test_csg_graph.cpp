/**
 * @file test_csg_graph.cpp
 * @brief Tests for tf::csg_graph and tf::make_csg_mesh.
 *
 * Builds two unit-cube meshes overlapping by 1x1x1, constructs a
 * csg_graph once, then extracts union / intersection / difference
 * meshes and checks each against the closed-form signed volume
 * (15 / 1 / 7), plus closedness and manifoldness.
 *
 * Also covers an `any_of({1})`-style sugar form to exercise the
 * initializer_list overload through the full pipeline.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include <algorithm>
#include <array>
#include <vector>

using graph_index_t = int;
using graph_real_t = float;
using graph_mesh_t = tf::polygons_buffer<graph_index_t, graph_real_t, 3, 3>;

namespace {

constexpr double graph_pi = tf::pi<double>;

auto translated(graph_mesh_t m, graph_real_t dx, graph_real_t dy,
                graph_real_t dz) -> graph_mesh_t {
  auto &pts = m.points_buffer();
  for (std::size_t i = 0; i < pts.size(); ++i) {
    auto p = pts[i];
    pts[i] = tf::point<graph_real_t, 3>{p[0] + dx, p[1] + dy, p[2] + dz};
  }
  return m;
}

template <typename Mesh>
void graph_check_solid(const Mesh &m, double expected_vol) {
  using Catch::Matchers::WithinAbs;
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
               WithinAbs(expected_vol, 1e-3));
}

} // namespace

TEST_CASE("csg_graph: two cubes stacked face-to-face (cross-tag coplanar)",
          "[csg][graph][coplanar]") {
  // Cube A: [0,1]^3. Cube B: [0,1]x[0,1]x[1,2]. Coplanar shared face
  // at z=1. Expected arrangement structure:
  //   - 2 components (one per cube, per-tag CCL invariant).
  //   - 1 bundle (NM-edge wedges around the shared face fuse them).
  //   - 3 domains: universe, inside_A, inside_B.
  // Union must produce a closed 1×1×2 box (volume 2).
  auto a = tf::make_box_mesh<graph_index_t>(graph_real_t(1), graph_real_t(1),
                                            graph_real_t(1));
  auto b = translated(tf::make_box_mesh<graph_index_t>(
                          graph_real_t(1), graph_real_t(1), graph_real_t(1)),
                      graph_real_t(0), graph_real_t(0), graph_real_t(1));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  SECTION("descriptor structure") {
    // 3 components: cube A splits into {body, surviving-top} because
    // the NM edges around the dead/survivor shared face (3 alive
    // faces meet) plus the per-tag CCL invariant isolate A's top
    // from A's walls. Cube B contributes 1 component (5 alive faces).
    REQUIRE(graph.labels().n_components() == graph_index_t(3));
    // One bundle: NM-edge wedges fuse the two cubes' bundles.
    REQUIRE(graph.descriptor().n_bundles == graph_index_t(1));
    // Three domains: universe, inside_A, inside_B.
    REQUIRE(graph.descriptor().n_domains == graph_index_t(3));

    // Per-tag invariant: each component belongs to one tag.
    auto tag_a = graph.descriptor().tag_of_component[0];
    auto tag_b = graph.descriptor().tag_of_component[2];
    CHECK(tag_a != tag_b);
    CHECK(graph.descriptor().tag_of_component[1] == tag_a);

    // The single bundle covers both tags.
    auto tags = graph.descriptor().bundle_to_tags[0];
    CHECK(tags.size() == 2);

    // Inclusion bits: each of {universe, inside_A, inside_B}
    // appears exactly once across the 3 domains; no "inside both".
    const auto wpd = graph.inclusion().words_per_domain;
    int counts[4] = {0, 0, 0, 0};
    for (graph_index_t d = 0; d < graph_index_t(3); ++d) {
      const auto bits = graph.inclusion().bits[d * wpd];
      counts[bits & 0x3]++;
    }
    CHECK(counts[0b00] == 1); // universe
    CHECK(counts[0b01] == 1); // inside A only
    CHECK(counts[0b10] == 1); // inside B only
    CHECK(counts[0b11] == 0); // no "inside both"
  }

  using Catch::Matchers::WithinAbs;
  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("union(A, B) = 1x1x2 box, shared face removed") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
    check(m, 2.0, 1e-3);
    // 1×1×2 box: 2 horizontal faces (2 tris each) + 4 vertical
    // walls (4 tris each, since each wall stacks A's + B's wall) =
    // 4 + 16 = 20 triangles. If the shared face at z=1 leaked
    // through it would be 20 + 2 = 22.
    REQUIRE(m.polygons().size() == std::size_t(20));
    // No triangle should lie entirely at z=1.
    auto polys = m.polygons();
    for (std::size_t f = 0; f < polys.size(); ++f) {
      auto poly = polys[f];
      bool all_at_z1 = true;
      for (int i = 0; i < 3; ++i)
        if (std::abs(poly[i][2] - graph_real_t(1)) > graph_real_t(1e-4))
          all_at_z1 = false;
      CHECK_FALSE(all_at_z1);
    }
  }
  SECTION("intersection(A, B) is empty (coplanar boundary touches only)") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(0.0, 1e-6));
  }
  SECTION("difference(A, B) = A") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)), 1.0, 1e-3);
  }
  SECTION("triangulation-only overloads (default intersect config)") {
    auto g0 = tf::test::build_range_csg_graph(
        tf::test::forms_range(forms), tf::test::no_sheets(),
        tf::triangulation_type::refined_cdt);
    check(tf::test::csg_mesh_of(g0, tf::csg::merge(0, 1)), 2.0, 1e-3);

    const int *none = nullptr;
    auto g1 = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                              tf::make_range(none, none),
                                              tf::triangulation_type::cdt);
    check(tf::test::csg_mesh_of(g1, tf::csg::merge(0, 1)), 2.0, 1e-3);
  }
}

TEST_CASE("csg_graph: A - B - C with two disjoint small spheres inside A",
          "[csg][graph][multi-bundle][nested-no-contact]") {
  // Three closed shells, three bundles, zero surface contact between
  // any of them. Big sphere A radius 5 at origin; small spheres B and
  // C radius 0.7 placed at ±(2, 0, 0) — both fully inside A, well
  // separated from each other. A - B - C must yield A with two
  // disjoint spherical holes; volume = vol_A - vol_B - vol_C.
  auto a = tf::make_sphere_mesh<graph_index_t>(graph_real_t(5), 48, 48);
  auto b_centered =
      tf::make_sphere_mesh<graph_index_t>(graph_real_t(0.7), 32, 32);
  auto b =
      translated(b_centered, graph_real_t(2), graph_real_t(0), graph_real_t(0));
  auto c = translated(b_centered, graph_real_t(-2), graph_real_t(0),
                      graph_real_t(0));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(3);
  operands.emplace_back(a);
  operands.emplace_back(b);
  operands.emplace_back(c);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  // No surface intersection between any pair, so the CSG output
  // re-emits original faces verbatim (some reversed for differences)
  // — no cuts, no triangulation, no tessellation error. Compare
  // against the input meshes' signed_volume directly for a tight
  // bound.
  const double sv_a = static_cast<double>(tf::signed_volume(a.polygons()));
  const double sv_b = static_cast<double>(tf::signed_volume(b.polygons()));
  const double sv_c = static_cast<double>(tf::signed_volume(c.polygons()));
  const double tol = 1e-2; // float32 summation noise across ~3k triangles

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("difference(A, merge(B, C)) = sv(A) - sv(B) - sv(C)") {
    check(tf::test::csg_mesh_of(graph,
                                tf::csg::difference(0, tf::csg::merge(1, 2))),
          sv_a - sv_b - sv_c, tol);
  }
  SECTION("sequential difference(difference(A, B), C) matches") {
    check(tf::test::csg_mesh_of(
              graph, tf::csg::difference(tf::csg::difference(0, 1), 2)),
          sv_a - sv_b - sv_c, tol);
  }
  SECTION("union(A, B, C) = sv(A) (B, C fully inside A)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::merge(0, tf::csg::merge(1, 2))),
          sv_a, tol);
  }
  SECTION("intersection(A, B) = sv(B) (B fully inside A)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)), sv_b, tol);
  }
}

TEST_CASE("csg_graph: outer sphere with two disjoint inner cubes",
          "[csg][graph][multi-bundle][shared-outer-env]") {
  // Three bundles total: outer sphere (bundle 0) and two inner cubes
  // (bundles 1, 2). Inner cubes do not touch each other and both sit
  // fully inside the sphere; their outer-env is the same domain (the
  // sphere's interior minus both cubes). Exercises two non-universe
  // raycast seeds whose target outer-env happens to coincide.
  auto outer = tf::make_sphere_mesh<graph_index_t>(graph_real_t(5), 48, 48);
  auto cube_a = tf::make_box_mesh<graph_index_t>(
      graph_real_t(1.5), graph_real_t(1.5), graph_real_t(1.5));
  // Place cubes on opposite sides of origin so they're definitely
  // disjoint and both inside the sphere.
  auto cubes = tf::concatenated(
      translated(cube_a, graph_real_t(-2), graph_real_t(0), graph_real_t(0))
          .polygons(),
      translated(cube_a, graph_real_t(2), graph_real_t(0), graph_real_t(0))
          .polygons());

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(outer);
  operands.emplace_back(cubes);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  const double vol_outer = (4.0 / 3.0) * graph_pi * 125.0;
  const double vol_two_cubes = 2.0 * std::pow(1.5, 3);
  const double tol = 3.0;

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("intersection = two cubes (both fully inside the sphere)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)),
          vol_two_cubes, 0.05);
  }
  SECTION("union = outer sphere (cubes are absorbed)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1)), vol_outer, tol);
  }
  SECTION("difference outer \\ cubes is sphere with two cubic holes") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
          vol_outer - vol_two_cubes, tol);
  }
  SECTION("difference cubes \\ sphere is empty") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::difference(1, 0));
    // Empty mesh: vacuously closed + manifold, signed_volume = 0.
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("csg_graph: four-deep concentric shells",
          "[csg][graph][multi-bundle][deep-nesting]") {
  // Four concentric spheres R=4, R=3, R=2, R=1. Four bundles, three
  // non-universe outer-envs. Each inner bundle's raycast crosses an
  // increasing number of outer shells.
  auto a = tf::make_sphere_mesh<graph_index_t>(graph_real_t(4), 48, 48);
  auto b = tf::make_sphere_mesh<graph_index_t>(graph_real_t(3), 48, 48);
  auto c = tf::make_sphere_mesh<graph_index_t>(graph_real_t(2), 48, 48);
  auto d = tf::make_sphere_mesh<graph_index_t>(graph_real_t(1), 32, 32);

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(4);
  operands.emplace_back(a);
  operands.emplace_back(b);
  operands.emplace_back(c);
  operands.emplace_back(d);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  const double vol_a = (4.0 / 3.0) * graph_pi * 64.0;
  const double vol_b = (4.0 / 3.0) * graph_pi * 27.0;
  const double vol_c = (4.0 / 3.0) * graph_pi * 8.0;
  const double vol_d = (4.0 / 3.0) * graph_pi * 1.0;
  const double tol = 2.5;

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("A \\ B = outer annulus only") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("B \\ C = middle annulus only") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(1, 2)),
          vol_b - vol_c, tol);
  }
  SECTION("C \\ D = innermost annulus") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(2, 3)),
          vol_c - vol_d, tol);
  }
  SECTION("A & D = D") {
    check(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 3)), vol_d,
          tol);
  }
  SECTION("any_of(A, B, C, D) = A") {
    check(tf::test::csg_mesh_of(graph, tf::csg::any_of({0, 1, 2, 3})), vol_a,
          tol);
  }
  SECTION("all_of(A, B, C, D) = D") {
    check(tf::test::csg_mesh_of(graph, tf::csg::all_of({0, 1, 2, 3})), vol_d,
          tol);
  }
}

TEST_CASE("csg_graph: three concentric spheres (universe -> A -> B -> C)",
          "[csg][graph][multi-bundle][nested-deep]") {
  // R=5, R=3, R=1 concentric spheres. All three bundles have non-
  // universe outer-envs except the outermost. Exercises the raycast
  // pass twice: middle's outer = annular(5,3), inner's outer =
  // annular(3,1). Inner's seed cast crosses both outer shells →
  // parity (1, 1, 0) on form bits (A, B, C self-skipped).
  auto a = tf::make_sphere_mesh<graph_index_t>(graph_real_t(5), 48, 48);
  auto b = tf::make_sphere_mesh<graph_index_t>(graph_real_t(3), 48, 48);
  auto c = tf::make_sphere_mesh<graph_index_t>(graph_real_t(1), 32, 32);

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(3);
  operands.emplace_back(a);
  operands.emplace_back(b);
  operands.emplace_back(c);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  const double vol_a = (4.0 / 3.0) * graph_pi * 125.0;
  const double vol_b = (4.0 / 3.0) * graph_pi * 27.0;
  const double vol_c = (4.0 / 3.0) * graph_pi * 1.0;
  const double tol = 2.5;

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("A | B | C = A") {
    check(tf::test::csg_mesh_of(graph, tf::csg::merge(0, tf::csg::merge(1, 2))),
          vol_a, tol);
  }
  SECTION("A & B & C = C") {
    check(tf::test::csg_mesh_of(
              graph, tf::csg::intersection(0, tf::csg::intersection(1, 2))),
          vol_c, tol);
  }
  SECTION("(A \\ B) is the outer annular shell") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("(B \\ C) is the middle annular shell") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(1, 2)),
          vol_b - vol_c, tol);
  }
  SECTION("symmetric_difference A xor C = (A \\ C) since C subset A") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 2)),
          vol_a - vol_c, tol);
  }
}

TEST_CASE("csg_graph: two side-by-side disjoint cubes vs a third cube touching one",
          "[csg][graph][multi-bundle][side-by-side]") {
  // Two cubes far apart (two disjoint bundles for form 0). Form 1 is
  // a single cube that fully overlaps cube0 of form 0 but is far from
  // cube1. Both bundles of form 0 have outer-env = universe (no
  // surface contact between them); the raycast for cube1's bundle
  // returns parity 0 for form 1 (no crossings). Exercises the
  // outer-env-equals-universe skip path inside multi-bundle.
  graph_real_t side = graph_real_t(2);
  auto cube0 = tf::make_box_mesh<graph_index_t>(side, side, side);
  auto cube1 = translated(tf::make_box_mesh<graph_index_t>(side, side, side),
                          graph_real_t(10), graph_real_t(0), graph_real_t(0));
  auto form0 = tf::concatenated(cube0.polygons(), cube1.polygons());
  auto form1 =
      tf::make_box_mesh<graph_index_t>(side, side, side); // overlaps cube0

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(form0);
  operands.emplace_back(form1);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("intersection(form0, form1) = cube0 (since form1 = cube0)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)), 8.0, 1e-3);
  }
  SECTION("union(form0, form1) = form0 (form1 already inside cube0)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1)), 16.0, 1e-3);
  }
  SECTION("difference(form0, form1) leaves only cube1 (cube0 fully removed)") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)), 8.0, 1e-3);
  }
}

TEST_CASE("csg_graph: nested-without-contact small sphere inside large sphere",
          "[csg][graph][multi-bundle]") {
  // Two concentric spheres, the inner radius 1 sits fully inside the
  // outer radius 3. No surface contact → two bundles, the inner
  // bundle's outer-env is bounded (inside the outer sphere) and
  // requires raycast seeding to flip the bit for form 0 (the outer
  // sphere).
  //   intersection(A, B) = B (volume = 4/3 π r³ = 4.18879…)
  //   union(A, B)        = A (volume = 4/3 π R³ = 113.0973…)
  //   A \ B              = A minus B (volume ≈ 108.9085)
  //   B \ A              = empty (no faces; signed_volume = 0)
  auto a = tf::make_sphere_mesh<graph_index_t>(graph_real_t(3), 48, 48);
  auto b = tf::make_sphere_mesh<graph_index_t>(graph_real_t(1), 32, 32);

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  const double vol_a = (4.0 / 3.0) * graph_pi * 27.0;
  const double vol_b = (4.0 / 3.0) * graph_pi * 1.0;
  const double tol = 1.0; // tessellation discretization on coarse spheres

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("intersection(A, B) = B") {
    check(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)), vol_b,
          tol);
  }
  SECTION("union(A, B) = A") {
    check(tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1)), vol_a, tol);
  }
  SECTION("difference(A, B) is the annular shell") {
    check(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("difference(B, A) is empty") {
    auto m = tf::test::csg_mesh_of(graph, tf::csg::difference(1, 0));
    // No surface should survive — B is fully inside A so B \ A is empty.
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("csg_graph: two-bundle form (two disjoint cubes) vs overlapping sphere",
          "[csg][graph][multi-bundle]") {
  // Form A = two disjoint unit cubes placed far apart (single form,
  // two arrangement bundles).
  // Form B = a sphere overlapping ONLY the first cube; it sits with
  // its centre at the centre of the first cube.
  // intersection(A, B) = sphere clipped to the first cube only;
  // the second cube is unaffected. Volume = volume of the cubic
  // intersection of the sphere of radius 1.2 with the cube [-1, 1]³ —
  // since r=1.2 > 1 along axes but the cube only extends ±1, the
  // intersection is the sphere's portion inside the cube. We compare
  // against the simpler quantity (union = (A) + (B) - intersection)
  // with closed-form A = 16 and B = 4/3 π 1.2³.
  graph_real_t cube_side = graph_real_t(2);
  graph_real_t r = graph_real_t(1.2);
  auto cube0 =
      tf::make_box_mesh<graph_index_t>(cube_side, cube_side, cube_side);
  auto cube1 = translated(
      tf::make_box_mesh<graph_index_t>(cube_side, cube_side, cube_side),
      graph_real_t(10), graph_real_t(0), graph_real_t(0));
  auto a = tf::concatenated(cube0.polygons(), cube1.polygons());
  auto b = tf::make_sphere_mesh<graph_index_t>(r, 48, 48);

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  const double vol_a = 16.0;
  const double vol_b = (4.0 / 3.0) * graph_pi * std::pow(double(r), 3);
  const double tol = 0.15;

  using Catch::Matchers::WithinAbs;

  SECTION("union(A, B) = A + (B - intersection)") {
    auto u = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
    auto i = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
    REQUIRE(tf::is_closed(u.polygons()));
    REQUIRE(tf::is_closed(i.polygons()));
    const double vol_u = double(tf::signed_volume(u.polygons()));
    const double vol_i = double(tf::signed_volume(i.polygons()));
    REQUIRE_THAT(vol_u, WithinAbs(vol_a + vol_b - vol_i, tol));
    REQUIRE(vol_i > 0.0);
  }
  SECTION("difference(A, B) = A - intersection") {
    auto d = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
    auto i = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
    REQUIRE(tf::is_closed(d.polygons()));
    const double vol_d = double(tf::signed_volume(d.polygons()));
    const double vol_i = double(tf::signed_volume(i.polygons()));
    REQUIRE_THAT(vol_d, WithinAbs(vol_a - vol_i, tol));
  }
}

TEST_CASE("csg_graph: two-cube booleans against closed-form volume",
          "[csg][graph]") {
  // Two unit cubes (side = 2) overlapping by 1x1x1:
  //   |A| = |B| = 8, |A intersection B| = 1, |A union B| = 15, |A \ B| = 7.
  auto a = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                            graph_real_t(2));
  auto b = translated(tf::make_box_mesh<graph_index_t>(
                          graph_real_t(2), graph_real_t(2), graph_real_t(2)),
                      graph_real_t(1), graph_real_t(1), graph_real_t(1));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  SECTION("union") {
    graph_check_solid(tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1)), 15.0);
  }
  SECTION("intersection") {
    graph_check_solid(tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1)),
                      1.0);
  }
  SECTION("difference") {
    graph_check_solid(tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1)),
                      7.0);
  }
  SECTION("any_of({1}) sugar equals difference(0, 1)") {
    auto m = tf::test::csg_mesh_of(
        graph, tf::csg::difference(0, tf::csg::any_of({1})));
    graph_check_solid(m, 7.0);
  }
}

TEST_CASE("make_csg_mesh return_source_ids: per-face tag + face provenance",
          "[csg][graph][source_ids]") {
  // Two unit cubes (side = 2) overlapping by 1x1x1, as above.
  auto a = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                            graph_real_t(2));
  auto b = translated(tf::make_box_mesh<graph_index_t>(
                          graph_real_t(2), graph_real_t(2), graph_real_t(2)),
                      graph_real_t(1), graph_real_t(1), graph_real_t(1));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto plain = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
  auto [mesh, tag_labels, face_labels] =
      tf::test::csg_mesh_with_source_ids_of(graph, tf::csg::merge(0, 1));

  // The mesh matches the plain build; labels run parallel to output faces.
  REQUIRE(mesh.faces().size() == plain.faces().size());
  REQUIRE(std::size_t(tag_labels.size()) == mesh.faces().size());
  REQUIRE(std::size_t(face_labels.size()) == mesh.faces().size());

  bool saw0 = false, saw1 = false;
  for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
    auto t = tag_labels[f];
    REQUIRE((t == graph_index_t(0) || t == graph_index_t(1)));
    saw0 = saw0 || (t == graph_index_t(0));
    saw1 = saw1 || (t == graph_index_t(1));
    // face_labels is an original face id within form t.
    auto fl = face_labels[f];
    REQUIRE(fl >= graph_index_t(0));
    REQUIRE(std::size_t(fl) < forms[std::size_t(t)].faces().size());
    // Order-independent correctness: the output face lies in the plane of
    // its claimed source face -- uncut faces are identical, cut faces are
    // triangulated within the source face's plane. (Frame is identity here.)
    auto src_plane = tf::make_plane(forms[std::size_t(t)][std::size_t(fl)]);
    for (auto v : mesh.polygons()[f]) {
      auto d = double(tf::distance(src_plane, v));
      REQUIRE(d < 1e-3);
      REQUIRE(d > -1e-3);
    }
  }
  // A union of two overlapping boxes keeps faces from both operands.
  REQUIRE(saw0);
  REQUIRE(saw1);
}

TEST_CASE("make_csg_mesh(graph): full arrangement matches make_mesh_arrangements",
          "[csg][graph][arrangement]") {
  // No expression -> the full arrangement surface, reusing the graph's
  // already-built intersection graph / face cuts (no pipeline rerun). Must
  // match the standalone make_mesh_arrangements pipeline face-for-face.
  auto a = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                            graph_real_t(2));
  auto b = translated(tf::make_box_mesh<graph_index_t>(
                          graph_real_t(2), graph_real_t(2), graph_real_t(2)),
                      graph_real_t(1), graph_real_t(1), graph_real_t(1));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  // Reference: the standalone arrangement pipeline.
  auto [ref_mesh, ref_tags, ref_faces] =
      tf::make_mesh_arrangements(tf::make_range(forms));

  // Graph path: same ig / fc, only the mesh is materialised.
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  auto arr = tf::make_csg_mesh(graph);

  REQUIRE(arr.faces().size() == ref_mesh.faces().size());
  REQUIRE(arr.points().size() == ref_mesh.points().size());
  REQUIRE(arr.faces().size() > 0);

  // Provenance overload: labels parallel to faces, valid, both operands seen.
  auto [arr2, tag_labels, face_labels] =
      tf::test::csg_mesh_with_source_ids_of(graph);
  REQUIRE(arr2.faces().size() == arr.faces().size());
  REQUIRE(std::size_t(tag_labels.size()) == arr2.faces().size());
  REQUIRE(std::size_t(face_labels.size()) == arr2.faces().size());

  bool saw0 = false, saw1 = false;
  for (std::size_t f = 0; f < arr2.faces().size(); ++f) {
    auto t = tag_labels[f];
    REQUIRE((t == graph_index_t(0) || t == graph_index_t(1)));
    saw0 = saw0 || (t == graph_index_t(0));
    saw1 = saw1 || (t == graph_index_t(1));
    auto fl = face_labels[f];
    REQUIRE(fl >= graph_index_t(0));
    REQUIRE(std::size_t(fl) < forms[std::size_t(t)].faces().size());
  }
  REQUIRE(saw0);
  REQUIRE(saw1);
}

TEST_CASE("make_csg_mesh return_index_map: point/face maps round-trip",
          "[csg][graph][index_map]") {
  auto a = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                            graph_real_t(2));
  auto b = translated(tf::make_box_mesh<graph_index_t>(
                          graph_real_t(2), graph_real_t(2), graph_real_t(2)),
                      graph_real_t(1), graph_real_t(1), graph_real_t(1));

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto check_map = [&](const auto &mesh, const auto &imap) {
    REQUIRE(std::size_t(imap.n_output_points) == mesh.points().size());
    REQUIRE(std::size_t(imap.face_tag_labels.size()) == mesh.faces().size());
    REQUIRE(std::size_t(imap.face_labels.size()) == mesh.faces().size());
    REQUIRE(std::size_t(imap.point_tag_labels.size()) == mesh.points().size());
    REQUIRE(std::size_t(imap.point_labels.size()) == mesh.points().size());

    // Face maps: valid (input form, input face).
    for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
      auto t = imap.face_tag_labels[f];
      REQUIRE((t >= graph_index_t(0) && std::size_t(t) < forms.size()));
      REQUIRE(std::size_t(imap.face_labels[f]) <
              forms[std::size_t(t)].faces().size());
    }

    // Point inverse: kept originals map to a valid input point; created
    // intersection points carry the end sentinel.
    for (graph_index_t o = graph_index_t(0); o < imap.n_output_points; ++o) {
      if (o < imap.n_original_points) {
        auto t = imap.point_tag_labels[std::size_t(o)];
        REQUIRE((t >= graph_index_t(0) && std::size_t(t) < forms.size()));
        REQUIRE(std::size_t(imap.point_labels[std::size_t(o)]) <
                forms[std::size_t(t)].points().size());
      } else {
        // Created point: tag axis ends at n_tags, point axis at n_output_points.
        REQUIRE(imap.point_tag_labels[std::size_t(o)] == imap.n_tags);
        REQUIRE(imap.point_labels[std::size_t(o)] == imap.n_output_points);
      }
    }

    // Forward/inverse round-trip: point_f[t][p] -> o, and the inverse at o
    // maps back to exactly (t, p). Input points dropped by a boolean carry the
    // end sentinel and are skipped.
    for (std::size_t t = 0; t < forms.size(); ++t) {
      auto blk = imap.point_f[t];
      REQUIRE(std::size_t(blk.size()) == forms[t].points().size());
      for (std::size_t p = 0; p < std::size_t(blk.size()); ++p) {
        auto o = blk[p];
        if (o != imap.n_output_points) {
          REQUIRE(o >= graph_index_t(0));
          REQUIRE(o < imap.n_output_points);
          REQUIRE(imap.point_tag_labels[std::size_t(o)] == graph_index_t(t));
          REQUIRE(imap.point_labels[std::size_t(o)] == graph_index_t(p));
        }
      }
    }
  };

  SECTION("full arrangement (no expression)") {
    auto [mesh, imap] = tf::test::csg_mesh_with_index_map_of(graph);
    check_map(mesh, imap);
  }

  SECTION("boolean expression") {
    auto plain = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
    auto [mesh, imap] =
        tf::test::csg_mesh_with_index_map_of(graph, tf::csg::merge(0, 1));
    REQUIRE(mesh.faces().size() == plain.faces().size());
    check_map(mesh, imap);
  }
}

TEST_CASE("csg_graph: two operands of different form types",
          "[csg][graph][pair]") {
  // The pair policy exists so the two operands need not share a C++
  // type — the shape every boolean call site has, where only one
  // operand carries a frame. Classification must read both through
  // `apply_to_form`, never through an indexable forms range.
  auto box = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                              graph_real_t(2));
  auto sphere = tf::make_sphere_mesh<graph_index_t>(graph_real_t(1.2), 12, 24);
  auto frame = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<graph_real_t, 3>{
          graph_real_t(0), graph_real_t(0), graph_real_t(0)}));

  auto bare = box.polygons();
  auto framed = sphere.polygons() | tf::tag(frame);
  static_assert(!std::is_same_v<decltype(bare), decltype(framed)>,
                "the operands must genuinely differ in type");

  auto graph = tf::make_csg_graph(bare, framed);
  REQUIRE(graph.arrangement().n_tags() == graph_index_t(2));
  REQUIRE(graph.descriptor().n_domains > graph_index_t(0));
  REQUIRE(graph.domain_volumes().size() ==
          std::size_t(graph.descriptor().n_domains));

  // A read that consumes the classification end to end.
  auto curves = tf::make_intersection_curves(graph);
  REQUIRE(curves.curves().size() > 0);

  // Same arrangement built homogeneously (both operands framed) must
  // agree on the domain count — the pair path is a storage choice,
  // not a semantic one.
  auto framed_box = box.polygons() | tf::tag(frame);
  std::vector<decltype(framed_box)> forms{framed_box, framed};
  auto homogeneous = tf::make_csg_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()));
  REQUIRE(homogeneous.descriptor().n_domains ==
          graph.descriptor().n_domains);

  // Extraction too: every op yields a valid solid, and the four
  // volumes satisfy the inclusion-exclusion identities, which no
  // per-operand bookkeeping slip could survive.
  auto volume_of = [](const auto &mesh) {
    auto p = mesh.polygons();
    REQUIRE(tf::is_closed(p));
    REQUIRE(tf::is_manifold(p));
    return double(tf::signed_volume(p));
  };
  const double v_union = volume_of(tf::make_csg_mesh(graph, tf::csg::op(0) |
                                                                tf::csg::op(1)));
  const double v_isect = volume_of(tf::make_csg_mesh(graph, tf::csg::op(0) &
                                                                tf::csg::op(1)));
  const double v_left = volume_of(tf::make_csg_mesh(graph, tf::csg::op(0) -
                                                               tf::csg::op(1)));
  const double v_right = volume_of(tf::make_csg_mesh(graph, tf::csg::op(1) -
                                                                tf::csg::op(0)));
  // the box is operand 0, and it is exactly 2x2x2
  REQUIRE_THAT(v_left + v_isect,
               Catch::Matchers::WithinRel(8.0, 1e-5));
  REQUIRE_THAT(v_union,
               Catch::Matchers::WithinRel(v_left + v_isect + v_right, 1e-5));
}

TEST_CASE("index map: uncut_faces separates whole faces from cut pieces",
          "[csg][graph][index_map]") {
  // Inside a tag's range the output face is still the entire input face;
  // outside it, every face is a piece of one. Consumers that carry
  // per-face data across an arrangement rely on exactly that split, so
  // check it against the geometry rather than against the producer.
  auto box = tf::make_box_mesh<graph_index_t>(graph_real_t(2), graph_real_t(2),
                                              graph_real_t(2));
  auto sphere = tf::make_sphere_mesh<graph_index_t>(graph_real_t(1.3), 12, 24);
  std::vector<decltype(box.polygons())> forms{box.polygons(),
                                              sphere.polygons()};
  auto rng = tf::make_range(forms.data(), forms.data() + forms.size());

  auto same_set = [](std::array<graph_index_t, 3> a,
                     std::array<graph_index_t, 3> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
  };
  auto check = [&](const auto &mesh, const auto &imap) {
    auto out = mesh.polygons();
    const graph_index_t n = graph_index_t(out.faces().size());
    std::vector<char> in_range(std::size_t(n), 0);
    for (graph_index_t t = 0; t < imap.n_tags; ++t) {
      REQUIRE(imap.uncut_faces[t][0] <= imap.uncut_faces[t][1]);
      REQUIRE(imap.uncut_faces[t][1] <= n);
      for (graph_index_t f = imap.uncut_faces[t][0]; f < imap.uncut_faces[t][1];
           ++f)
        in_range[std::size_t(f)] = 1;
    }
    for (graph_index_t f = 0; f < n; ++f) {
      const graph_index_t t = imap.face_tag_labels[std::size_t(f)];
      const graph_index_t orig = imap.face_labels[std::size_t(f)];
      auto fwd = imap.point_f[t];
      std::array<graph_index_t, 3> mapped{}, emitted{};
      for (int c = 0; c < 3; ++c) {
        mapped[std::size_t(c)] =
            fwd[forms[std::size_t(t)].faces()[orig][std::size_t(c)]];
        emitted[std::size_t(c)] = out.faces()[f][std::size_t(c)];
      }
      REQUIRE(same_set(mapped, emitted) == bool(in_range[std::size_t(f)]));
    }
  };

  SECTION("full arrangement") {
    auto [mesh, imap] = tf::make_mesh_arrangements(rng, tf::arrangement_config{},
                                                   tf::return_index_map);
    check(mesh, imap);
  }
  SECTION("boolean selections") {
    auto graph = tf::make_csg_graph(rng);
    for (auto e : {tf::csg::op(0) | tf::csg::op(1), tf::csg::op(0) & tf::csg::op(1),
                   tf::csg::op(0) - tf::csg::op(1), tf::csg::op(1) - tf::csg::op(0)}) {
      auto [mesh, imap] = tf::make_csg_mesh(graph, e, tf::return_index_map);
      check(mesh, imap);
    }
  }
}

// The bundle axis IS the component axis when no radial fan exists — the
// branch @ref tf::csg::graph::make_arrangement_descriptor takes for operands
// that nowhere meet. Every pass keyed on bundles then runs over the whole
// component space, so this is the branch that decides whether a block-local
// dense array over that axis is affordable; the grid holds it wide.
TEST_CASE("csg_graph: separated solids make the bundle axis the component axis",
          "[csg][graph][bundles]") {
  const int per_side = 8;
  const graph_real_t pitch = 4;
  auto unit = tf::make_box_mesh<graph_index_t>(graph_real_t(1), graph_real_t(1),
                                               graph_real_t(1));
  auto grid = [&](graph_real_t offset) {
    graph_mesh_t out;
    auto &pts = out.points_buffer();
    auto &faces = out.faces_buffer();
    for (int i = 0; i < per_side; ++i)
      for (int j = 0; j < per_side; ++j)
        for (int k = 0; k < per_side; ++k) {
          const auto base = graph_index_t(pts.size());
          for (auto p : unit.points())
            pts.push_back(tf::point<graph_real_t, 3>{
                p[0] + graph_real_t(i) * pitch + offset,
                p[1] + graph_real_t(j) * pitch,
                p[2] + graph_real_t(k) * pitch});
          for (auto f : unit.faces())
            for (int c = 0; c < 3; ++c)
              faces.data_buffer().push_back(base +
                                            graph_index_t(f[std::size_t(c)]));
        }
    return out;
  };
  auto a = grid(graph_real_t(0));
  auto b = grid(graph_real_t(2 * per_side) * pitch);
  const auto n_solids = graph_index_t(2 * per_side * per_side * per_side);

  std::vector<tf::test::tagged_operand<graph_index_t, graph_real_t>> operands;
  operands.reserve(2);
  operands.emplace_back(a);
  operands.emplace_back(b);
  auto forms = tf::test::tagged_forms(operands);

  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  // nothing meets, so no piece carries a fan and the descriptor names every
  // component its own bundle
  REQUIRE(graph.descriptor().fans.pieces.size() == 0);
  REQUIRE(graph.descriptor().n_bundles == graph.labels().n_components());
  REQUIRE(graph.descriptor().n_bundles == n_solids);

  auto [cells, ids] = tf::test::csg_domains_of(graph);
  REQUIRE(graph_index_t(ids.size()) == n_solids);
  for (std::size_t d = 0; d < std::size_t(ids.size()); ++d)
    graph_check_solid(cells[d], 1.0);
}
