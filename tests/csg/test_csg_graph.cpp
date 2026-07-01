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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/trueform.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>

#include <vector>

using Index = int;
using Real = float;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

namespace {

constexpr double pi = tf::pi<double>;

auto translated(mesh_t m, Real dx, Real dy, Real dz) -> mesh_t {
  auto &pts = m.points_buffer();
  for (std::size_t i = 0; i < pts.size(); ++i) {
    auto p = pts[i];
    pts[i] = tf::point<Real, 3>{p[0] + dx, p[1] + dy, p[2] + dz};
  }
  return m;
}

template <typename Mesh>
void check_solid(const Mesh &m, double expected_vol) {
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
  auto a = tf::make_box_mesh<Index>(Real(1), Real(1), Real(1));
  auto b = translated(tf::make_box_mesh<Index>(Real(1), Real(1), Real(1)),
                       Real(0), Real(0), Real(1));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  SECTION("descriptor structure") {
    // 3 components: cube A splits into {body, surviving-top} because
    // the NM edges around the dead/survivor shared face (3 alive
    // faces meet) plus the per-tag CCL invariant isolate A's top
    // from A's walls. Cube B contributes 1 component (5 alive faces).
    REQUIRE(graph.arrangement().n_components() == Index(3));
    // One bundle: NM-edge wedges fuse the two cubes' bundles.
    REQUIRE(graph.descriptor().n_bundles == Index(1));
    // Three domains: universe, inside_A, inside_B.
    REQUIRE(graph.descriptor().n_domains == Index(3));

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
    for (Index d = 0; d < Index(3); ++d) {
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
    auto m = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
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
        if (std::abs(poly[i][2] - Real(1)) > Real(1e-4))
          all_at_z1 = false;
      CHECK_FALSE(all_at_z1);
    }
  }
  SECTION("intersection(A, B) is empty (coplanar boundary touches only)") {
    auto m = tf::make_csg_mesh(graph, tf::csg::intersection(0, 1));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(0.0, 1e-6));
  }
  SECTION("difference(A, B) = A") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)), 1.0, 1e-3);
  }
}

TEST_CASE("csg_graph: A - B - C with two disjoint small spheres inside A",
          "[csg][graph][multi-bundle][nested-no-contact]") {
  // Three closed shells, three bundles, zero surface contact between
  // any of them. Big sphere A radius 5 at origin; small spheres B and
  // C radius 0.7 placed at ±(2, 0, 0) — both fully inside A, well
  // separated from each other. A - B - C must yield A with two
  // disjoint spherical holes; volume = vol_A - vol_B - vol_C.
  auto a = tf::make_sphere_mesh<Index>(Real(5), 48, 48);
  auto b_centered = tf::make_sphere_mesh<Index>(Real(0.7), 32, 32);
  auto b = translated(b_centered, Real(2), Real(0), Real(0));
  auto c = translated(b_centered, Real(-2), Real(0), Real(0));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(3);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));
  forms.push_back(c.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

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
    check(tf::make_csg_mesh(graph,
                              tf::csg::difference(0, tf::csg::merge(1, 2))),
          sv_a - sv_b - sv_c, tol);
  }
  SECTION("sequential difference(difference(A, B), C) matches") {
    check(tf::make_csg_mesh(
              graph, tf::csg::difference(tf::csg::difference(0, 1), 2)),
          sv_a - sv_b - sv_c, tol);
  }
  SECTION("union(A, B, C) = sv(A) (B, C fully inside A)") {
    check(tf::make_csg_mesh(graph,
                              tf::csg::merge(0, tf::csg::merge(1, 2))),
          sv_a, tol);
  }
  SECTION("intersection(A, B) = sv(B) (B fully inside A)") {
    check(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)),
          sv_b, tol);
  }
}

TEST_CASE("csg_graph: outer sphere with two disjoint inner cubes",
          "[csg][graph][multi-bundle][shared-outer-env]") {
  // Three bundles total: outer sphere (bundle 0) and two inner cubes
  // (bundles 1, 2). Inner cubes do not touch each other and both sit
  // fully inside the sphere; their outer-env is the same domain (the
  // sphere's interior minus both cubes). Exercises two non-universe
  // raycast seeds whose target outer-env happens to coincide.
  auto outer = tf::make_sphere_mesh<Index>(Real(5), 48, 48);
  auto cube_a = tf::make_box_mesh<Index>(Real(1.5), Real(1.5), Real(1.5));
  // Place cubes on opposite sides of origin so they're definitely
  // disjoint and both inside the sphere.
  auto cubes = tf::concatenated(
      translated(cube_a, Real(-2), Real(0), Real(0)).polygons(),
      translated(cube_a, Real(2), Real(0), Real(0)).polygons());

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(outer.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(outer.polygons() | tf::tag(frame_id));
  forms.push_back(cubes.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  const double vol_outer = (4.0 / 3.0) * pi * 125.0;
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
    check(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)),
          vol_two_cubes, 0.05);
  }
  SECTION("union = outer sphere (cubes are absorbed)") {
    check(tf::make_csg_mesh(graph, tf::csg::merge(0, 1)), vol_outer, tol);
  }
  SECTION("difference outer \\ cubes is sphere with two cubic holes") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)),
          vol_outer - vol_two_cubes, tol);
  }
  SECTION("difference cubes \\ sphere is empty") {
    auto m = tf::make_csg_mesh(graph, tf::csg::difference(1, 0));
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
  auto a = tf::make_sphere_mesh<Index>(Real(4), 48, 48);
  auto b = tf::make_sphere_mesh<Index>(Real(3), 48, 48);
  auto c = tf::make_sphere_mesh<Index>(Real(2), 48, 48);
  auto d = tf::make_sphere_mesh<Index>(Real(1), 32, 32);

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(4);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));
  forms.push_back(c.polygons() | tf::tag(frame_id));
  forms.push_back(d.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  const double vol_a = (4.0 / 3.0) * pi * 64.0;
  const double vol_b = (4.0 / 3.0) * pi * 27.0;
  const double vol_c = (4.0 / 3.0) * pi * 8.0;
  const double vol_d = (4.0 / 3.0) * pi * 1.0;
  const double tol = 2.5;

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("A \\ B = outer annulus only") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("B \\ C = middle annulus only") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(1, 2)),
          vol_b - vol_c, tol);
  }
  SECTION("C \\ D = innermost annulus") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(2, 3)),
          vol_c - vol_d, tol);
  }
  SECTION("A & D = D") {
    check(tf::make_csg_mesh(graph, tf::csg::intersection(0, 3)), vol_d, tol);
  }
  SECTION("any_of(A, B, C, D) = A") {
    check(tf::make_csg_mesh(graph, tf::csg::any_of({0, 1, 2, 3})), vol_a,
          tol);
  }
  SECTION("all_of(A, B, C, D) = D") {
    check(tf::make_csg_mesh(graph, tf::csg::all_of({0, 1, 2, 3})), vol_d,
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
  auto a = tf::make_sphere_mesh<Index>(Real(5), 48, 48);
  auto b = tf::make_sphere_mesh<Index>(Real(3), 48, 48);
  auto c = tf::make_sphere_mesh<Index>(Real(1), 32, 32);

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(3);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));
  forms.push_back(c.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  const double vol_a = (4.0 / 3.0) * pi * 125.0;
  const double vol_b = (4.0 / 3.0) * pi * 27.0;
  const double vol_c = (4.0 / 3.0) * pi * 1.0;
  const double tol = 2.5;

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("A | B | C = A") {
    check(tf::make_csg_mesh(graph, tf::csg::merge(0, tf::csg::merge(1, 2))),
          vol_a, tol);
  }
  SECTION("A & B & C = C") {
    check(tf::make_csg_mesh(
              graph, tf::csg::intersection(0, tf::csg::intersection(1, 2))),
          vol_c, tol);
  }
  SECTION("(A \\ B) is the outer annular shell") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("(B \\ C) is the middle annular shell") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(1, 2)),
          vol_b - vol_c, tol);
  }
  SECTION("symmetric_difference A xor C = (A \\ C) since C subset A") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 2)),
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
  Real side = Real(2);
  auto cube0 = tf::make_box_mesh<Index>(side, side, side);
  auto cube1 = translated(tf::make_box_mesh<Index>(side, side, side),
                           Real(10), Real(0), Real(0));
  auto form0 = tf::concatenated(cube0.polygons(), cube1.polygons());
  auto form1 = tf::make_box_mesh<Index>(side, side, side); // overlaps cube0

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(form0.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(form0.polygons() | tf::tag(frame_id));
  forms.push_back(form1.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("intersection(form0, form1) = cube0 (since form1 = cube0)") {
    check(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)), 8.0, 1e-3);
  }
  SECTION("union(form0, form1) = form0 (form1 already inside cube0)") {
    check(tf::make_csg_mesh(graph, tf::csg::merge(0, 1)), 16.0, 1e-3);
  }
  SECTION("difference(form0, form1) leaves only cube1 (cube0 fully removed)") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)), 8.0, 1e-3);
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
  auto a = tf::make_sphere_mesh<Index>(Real(3), 48, 48);
  auto b = tf::make_sphere_mesh<Index>(Real(1), 32, 32);

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  const double vol_a = (4.0 / 3.0) * pi * 27.0;
  const double vol_b = (4.0 / 3.0) * pi * 1.0;
  const double tol = 1.0; // tessellation discretization on coarse spheres

  using Catch::Matchers::WithinAbs;

  auto check = [](const auto &m, double expected, double t) {
    REQUIRE(tf::is_closed(m.polygons()));
    REQUIRE(tf::is_manifold(m.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(m.polygons())),
                 WithinAbs(expected, t));
  };

  SECTION("intersection(A, B) = B") {
    check(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)), vol_b, tol);
  }
  SECTION("union(A, B) = A") {
    check(tf::make_csg_mesh(graph, tf::csg::merge(0, 1)), vol_a, tol);
  }
  SECTION("difference(A, B) is the annular shell") {
    check(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)),
          vol_a - vol_b, tol);
  }
  SECTION("difference(B, A) is empty") {
    auto m = tf::make_csg_mesh(graph, tf::csg::difference(1, 0));
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
  Real cube_side = Real(2);
  Real r = Real(1.2);
  auto cube0 = tf::make_box_mesh<Index>(cube_side, cube_side, cube_side);
  auto cube1 = translated(tf::make_box_mesh<Index>(cube_side, cube_side,
                                                    cube_side),
                           Real(10), Real(0), Real(0));
  auto a = tf::concatenated(cube0.polygons(), cube1.polygons());
  auto b = tf::make_sphere_mesh<Index>(r, 48, 48);

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  const double vol_a = 16.0;
  const double vol_b = (4.0 / 3.0) * pi * std::pow(double(r), 3);
  const double tol = 0.15;

  using Catch::Matchers::WithinAbs;

  SECTION("union(A, B) = A + (B - intersection)") {
    auto u = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
    auto i = tf::make_csg_mesh(graph, tf::csg::intersection(0, 1));
    REQUIRE(tf::is_closed(u.polygons()));
    REQUIRE(tf::is_closed(i.polygons()));
    const double vol_u = double(tf::signed_volume(u.polygons()));
    const double vol_i = double(tf::signed_volume(i.polygons()));
    REQUIRE_THAT(vol_u, WithinAbs(vol_a + vol_b - vol_i, tol));
    REQUIRE(vol_i > 0.0);
  }
  SECTION("difference(A, B) = A - intersection") {
    auto d = tf::make_csg_mesh(graph, tf::csg::difference(0, 1));
    auto i = tf::make_csg_mesh(graph, tf::csg::intersection(0, 1));
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
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = translated(tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
                       Real(1), Real(1), Real(1));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  SECTION("union") {
    check_solid(tf::make_csg_mesh(graph, tf::csg::merge(0, 1)), 15.0);
  }
  SECTION("intersection") {
    check_solid(tf::make_csg_mesh(graph, tf::csg::intersection(0, 1)), 1.0);
  }
  SECTION("difference") {
    check_solid(tf::make_csg_mesh(graph, tf::csg::difference(0, 1)), 7.0);
  }
  SECTION("any_of({1}) sugar equals difference(0, 1)") {
    auto m = tf::make_csg_mesh(graph,
                                tf::csg::difference(0, tf::csg::any_of({1})));
    check_solid(m, 7.0);
  }
}

TEST_CASE("make_csg_mesh return_source_ids: per-face tag + face provenance",
          "[csg][graph][source_ids]") {
  // Two unit cubes (side = 2) overlapping by 1x1x1, as above.
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = translated(tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
                       Real(1), Real(1), Real(1));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto plain = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
  auto [mesh, tag_labels, face_labels] =
      tf::make_csg_mesh(graph, tf::csg::merge(0, 1), tf::return_source_ids);

  // The mesh matches the plain build; labels run parallel to output faces.
  REQUIRE(mesh.faces().size() == plain.faces().size());
  REQUIRE(std::size_t(tag_labels.size()) == mesh.faces().size());
  REQUIRE(std::size_t(face_labels.size()) == mesh.faces().size());

  bool saw0 = false, saw1 = false;
  for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
    auto t = tag_labels[f];
    REQUIRE((t == Index(0) || t == Index(1)));
    saw0 = saw0 || (t == Index(0));
    saw1 = saw1 || (t == Index(1));
    // face_labels is an original face id within form t.
    auto fl = face_labels[f];
    REQUIRE(fl >= Index(0));
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
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = translated(tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
                       Real(1), Real(1), Real(1));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));

  // Reference: the standalone arrangement pipeline.
  auto [ref_mesh, ref_tags, ref_faces] =
      tf::make_mesh_arrangements(tf::make_range(forms));

  // Graph path: same ig / fc, only the mesh is materialised.
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  auto arr = tf::make_csg_mesh(graph);

  REQUIRE(arr.faces().size() == ref_mesh.faces().size());
  REQUIRE(arr.points().size() == ref_mesh.points().size());
  REQUIRE(arr.faces().size() > 0);

  // Provenance overload: labels parallel to faces, valid, both operands seen.
  auto [arr2, tag_labels, face_labels] =
      tf::make_csg_mesh(graph, tf::return_source_ids);
  REQUIRE(arr2.faces().size() == arr.faces().size());
  REQUIRE(std::size_t(tag_labels.size()) == arr2.faces().size());
  REQUIRE(std::size_t(face_labels.size()) == arr2.faces().size());

  bool saw0 = false, saw1 = false;
  for (std::size_t f = 0; f < arr2.faces().size(); ++f) {
    auto t = tag_labels[f];
    REQUIRE((t == Index(0) || t == Index(1)));
    saw0 = saw0 || (t == Index(0));
    saw1 = saw1 || (t == Index(1));
    auto fl = face_labels[f];
    REQUIRE(fl >= Index(0));
    REQUIRE(std::size_t(fl) < forms[std::size_t(t)].faces().size());
  }
  REQUIRE(saw0);
  REQUIRE(saw1);
}

TEST_CASE("make_csg_mesh return_index_map: point/face maps round-trip",
          "[csg][graph][index_map]") {
  auto a = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto b = translated(tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
                       Real(1), Real(1), Real(1));

  auto frame_id = tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
  using form_t = decltype(a.polygons() | tf::tag(frame_id));
  std::vector<form_t> forms;
  forms.reserve(2);
  forms.push_back(a.polygons() | tf::tag(frame_id));
  forms.push_back(b.polygons() | tf::tag(frame_id));
  auto graph = tf::make_csg_graph(tf::make_range(forms));

  auto check_map = [&](const auto &mesh, const auto &imap) {
    REQUIRE(std::size_t(imap.n_output_points) == mesh.points().size());
    REQUIRE(std::size_t(imap.face_tag_labels.size()) == mesh.faces().size());
    REQUIRE(std::size_t(imap.face_labels.size()) == mesh.faces().size());
    REQUIRE(std::size_t(imap.point_tag_labels.size()) == mesh.points().size());
    REQUIRE(std::size_t(imap.point_labels.size()) == mesh.points().size());

    // Face maps: valid (input form, input face).
    for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
      auto t = imap.face_tag_labels[f];
      REQUIRE((t >= Index(0) && std::size_t(t) < forms.size()));
      REQUIRE(std::size_t(imap.face_labels[f]) <
              forms[std::size_t(t)].faces().size());
    }

    // Point inverse: kept originals map to a valid input point; created
    // intersection points carry the end sentinel.
    for (Index o = Index(0); o < imap.n_output_points; ++o) {
      if (o < imap.n_original_points) {
        auto t = imap.point_tag_labels[std::size_t(o)];
        REQUIRE((t >= Index(0) && std::size_t(t) < forms.size()));
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
          REQUIRE(o >= Index(0));
          REQUIRE(o < imap.n_output_points);
          REQUIRE(imap.point_tag_labels[std::size_t(o)] == Index(t));
          REQUIRE(imap.point_labels[std::size_t(o)] == Index(p));
        }
      }
    }
  };

  SECTION("full arrangement (no expression)") {
    auto [mesh, imap] = tf::make_csg_mesh(graph, tf::return_index_map);
    check_map(mesh, imap);
  }

  SECTION("boolean expression") {
    auto plain = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
    auto [mesh, imap] =
        tf::make_csg_mesh(graph, tf::csg::merge(0, 1), tf::return_index_map);
    REQUIRE(mesh.faces().size() == plain.faces().size());
    check_map(mesh, imap);
  }
}
