/**
 * @file test_mesh_arrangements.cpp
 * @brief Tests for make_mesh_arrangements with N meshes
 *
 * Verifies structural properties of mesh arrangements:
 * - No degenerate triangles
 * - All tags present
 * - Intersection curves match non-manifold paths
 * - Curve count and endpoint count for known configurations
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "type_traits.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <set>
#include <trueform/trueform.hpp>

#include "arrangement_builders.hpp"
#include "arrangement_readers.hpp"
#include "tagged_operand.hpp"
#include <vector>

template <typename index_t, typename Paths>
auto get_sorted_sizes(const Paths &paths) -> std::vector<std::size_t> {
  std::vector<std::size_t> sizes;
  for (auto path : paths)
    sizes.push_back(path.size());
  std::sort(sizes.begin(), sizes.end());
  return sizes;
}

template <typename index_t, typename Paths>
auto count_open_endpoints(const Paths &paths) -> int {
  std::set<index_t> eps;
  for (auto path : paths) {
    if (path.size() < 2)
      continue;
    if (path[0] != path[path.size() - 1]) {
      eps.insert(path[0]);
      eps.insert(path[path.size() - 1]);
    }
  }
  return static_cast<int>(eps.size());
}

template <typename Mesh>
auto mesh_arrangements_count_degenerate(const Mesh &mesh) -> int {
  int n = 0;
  for (auto face : mesh.faces())
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
      ++n;
  return n;
}

template <typename Mesh> auto total_polygon_area(const Mesh &mesh) -> double {
  double sum = 0;
  for (auto face : mesh.polygons())
    for (std::size_t k = 2; k < face.size(); ++k) {
      const auto a = face[0], b = face[k - 1], c = face[k];
      const double ux = double(b[0]) - double(a[0]);
      const double uy = double(b[1]) - double(a[1]);
      const double uz = double(b[2]) - double(a[2]);
      const double vx = double(c[0]) - double(a[0]);
      const double vy = double(c[1]) - double(a[1]);
      const double vz = double(c[2]) - double(a[2]);
      const double nx = uy * vz - uz * vy;
      const double ny = uz * vx - ux * vz;
      const double nz = ux * vy - uy * vx;
      sum += 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);
    }
  return sum;
}

/// A hexagon of area 36 in z = 0 whose corner cycle starts at `shift`. The
/// corners 0, 1, 2 of the unshifted cycle are collinear — a subdivided side —
/// and that run stays collinear on the lattice because it varies in x alone.
template <typename Index, typename Real>
auto make_collinear_run_hexagon(int shift)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  const std::array<std::array<Real, 3>, 6> corners{{
      {{Real(-4), Real(-2), Real(0)}},
      {{Real(0), Real(-2), Real(0)}},
      {{Real(4), Real(-2), Real(0)}},
      {{Real(4), Real(2), Real(0)}},
      {{Real(0), Real(3), Real(0)}},
      {{Real(-4), Real(2), Real(0)}},
  }};
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> mesh;
  for (const auto &corner : corners)
    mesh.points_buffer().emplace_back(corner[0], corner[1], corner[2]);
  tf::buffer<Index> face;
  for (int k = 0; k < 6; ++k)
    face.push_back(Index((k + shift) % 6));
  mesh.faces_buffer().push_back(tf::make_range(face));
  return mesh;
}

/// A blade of area 72 standing in y = 0, crossing that hexagon.
template <typename Index, typename Real>
auto make_crossing_blade() -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  mesh.points_buffer().emplace_back(Real(-6), Real(0), Real(-3));
  mesh.points_buffer().emplace_back(Real(6), Real(0), Real(-3));
  mesh.points_buffer().emplace_back(Real(6), Real(0), Real(3));
  mesh.points_buffer().emplace_back(Real(-6), Real(0), Real(3));
  mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
  mesh.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
  return mesh;
}

TEMPLATE_TEST_CASE("mesh_arrangements_3_spheres", "[arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s2 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(-0.5), real_t(0), real_t(0)}));
  auto f2 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0), real_t(0.5), real_t(0)}));

  std::vector<tf::test::tagged_operand<index_t, real_t>> operands;
  operands.reserve(3);
  operands.push_back(tf::test::make_tagged_operand(
      s0, tf::transformation<real_t, 3>(f0.transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      s1, tf::transformation<real_t, 3>(f1.transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      s2, tf::transformation<real_t, 3>(f2.transformation())));
  auto forms = tf::test::tagged_forms(operands);

  auto [mesh, tag_labels, face_labels, curves] =
      tf::test::mesh_arrangements_with_curves_of(tf::test::forms_range(forms),
                                                 tf::intersect_config{});

  REQUIRE(mesh_arrangements_count_degenerate(mesh) == 0);

  int tag_counts[3] = {};
  for (auto t : tag_labels)
    tag_counts[t]++;
  REQUIRE(tag_counts[0] > 0);
  REQUIRE(tag_counts[1] > 0);
  REQUIRE(tag_counts[2] > 0);

  auto nmedges = tf::make_non_manifold_edges(mesh.polygons());
  auto nm_paths = tf::connect_edges_to_paths(tf::make_edges(nmedges));
  auto ig_sizes = get_sorted_sizes<index_t>(curves.paths());
  auto nm_sizes = get_sorted_sizes<index_t>(nm_paths);
  REQUIRE(ig_sizes.size() == nm_sizes.size());
  REQUIRE(ig_sizes == nm_sizes);

  // 3 pairwise intersections split at 2 triple points → 6 curves, 2 endpoints
  REQUIRE(curves.paths().size() == 6);
  REQUIRE(count_open_endpoints<index_t>(curves.paths()) == 2);

  // Closed inputs → closed output
  REQUIRE(tf::make_boundary_edges(mesh.polygons()).size() == 0);
}

TEMPLATE_TEST_CASE("mesh_arrangements_2_spheres", "[arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 25, 25);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(0.8), 20, 20);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

  auto op0 = tf::test::make_tagged_operand(s0);
  auto op1 = tf::test::make_tagged_operand(
      s1, tf::transformation<real_t, 3>(frame.transformation()));
  std::vector<tf::test::form_t<index_t, real_t, 3>> forms{op0.form(),
                                                          op1.form()};
  auto [mesh, tag_labels, face_labels, curves] =
      tf::test::mesh_arrangements_with_curves_of(tf::test::forms_range(forms),
                                                 tf::intersect_config{});

  REQUIRE(mesh_arrangements_count_degenerate(mesh) == 0);

  int tag_counts[2] = {};
  for (auto t : tag_labels)
    tag_counts[t]++;
  REQUIRE(tag_counts[0] > 0);
  REQUIRE(tag_counts[1] > 0);

  auto nmedges = tf::make_non_manifold_edges(mesh.polygons());
  auto nm_paths = tf::connect_edges_to_paths(tf::make_edges(nmedges));
  auto ig_sizes = get_sorted_sizes<index_t>(curves.paths());
  auto nm_sizes = get_sorted_sizes<index_t>(nm_paths);
  REQUIRE(ig_sizes.size() == nm_sizes.size());
  REQUIRE(ig_sizes == nm_sizes);

  // 2 overlapping spheres → 1 closed intersection curve, 0 endpoints
  REQUIRE(curves.paths().size() == 1);
  REQUIRE(count_open_endpoints<index_t>(curves.paths()) == 0);

  // Closed inputs → closed output
  REQUIRE(tf::make_boundary_edges(mesh.polygons()).size() == 0);
}

TEST_CASE("mesh_arrangements_box_cylinder_sphere", "[arrangements]") {
  using index_t = int;
  using real_t = float;

  auto box = tf::make_box_mesh<index_t>(real_t(5), real_t(2), real_t(5));
  auto cylinder = tf::make_cylinder_mesh<index_t>(real_t(2), real_t(10), 50);
  auto sphere = tf::make_sphere_mesh<index_t>(real_t(3), 20, 20);

  std::vector<decltype(tf::test::make_tagged_operand(box))> operands;
  operands.reserve(3);
  operands.push_back(tf::test::make_tagged_operand(box));
  operands.push_back(tf::test::make_tagged_operand(cylinder));
  operands.push_back(tf::test::make_tagged_operand(sphere));
  auto forms = tf::test::tagged_forms(operands);

  auto [mesh, tag_labels, face_labels, curves] =
      tf::test::mesh_arrangements_with_curves_of(
          tf::test::forms_range(forms),
          tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_crossing_contours});

  REQUIRE(mesh_arrangements_count_degenerate(mesh) == 0);

  int tag_counts[3] = {};
  for (auto t : tag_labels)
    tag_counts[t]++;
  REQUIRE(tag_counts[0] > 0);
  REQUIRE(tag_counts[1] > 0);
  REQUIRE(tag_counts[2] > 0);

  auto nmedges = tf::make_non_manifold_edges(mesh.polygons());
  auto nm_paths = tf::connect_edges_to_paths(tf::make_edges(nmedges));
  auto ig_sizes = get_sorted_sizes<index_t>(curves.paths());
  auto nm_sizes = get_sorted_sizes<index_t>(nm_paths);
  REQUIRE(ig_sizes.size() == nm_sizes.size());
  REQUIRE(ig_sizes == nm_sizes);

  // 3 pairwise intersections with 8 junction points
  REQUIRE(count_open_endpoints<index_t>(curves.paths()) == 8);

  // Closed inputs - closed output (the earcut coincident-vertex fix)
  REQUIRE(tf::make_boundary_edges(mesh.polygons()).size() == 0);
}

TEST_CASE("mesh_arrangements_tolerance_parallel_sheets",
          "[arrangements][tolerance]") {
  // Two parallel unit quads offset by `d` along z. The box_plane case
  // above only exercises the band inside an exact incidence (the cube's
  // walls cross the fin's plane); here NEITHER face crosses the other's
  // plane, so the pairing itself must honor the band: within it the sheets
  // are one feature, beyond it two.
  using index_t = int;
  using real_t = float;

  auto make_sheet = [](real_t z) {
    tf::polygons_buffer<index_t, real_t, 3, 3> sheet;
    sheet.points_buffer().allocate(4);
    sheet.faces_buffer().allocate(2);
    sheet.points_buffer()[0] = tf::point<real_t, 3>{real_t(0), real_t(0), z};
    sheet.points_buffer()[1] = tf::point<real_t, 3>{real_t(1), real_t(0), z};
    sheet.points_buffer()[2] = tf::point<real_t, 3>{real_t(1), real_t(1), z};
    sheet.points_buffer()[3] = tf::point<real_t, 3>{real_t(0), real_t(1), z};
    sheet.faces_buffer()[0] = std::array<index_t, 3>{0, 1, 2};
    sheet.faces_buffer()[1] = std::array<index_t, 3>{0, 2, 3};
    return sheet;
  };
  const double tolerance = 1e-3;
  const auto mode = tf::intersect_mode::primitives |
                    tf::intersect_mode::resolve_contours |
                    tf::intersect_mode::within;
  auto arrange = [&](real_t d) {
    auto a = make_sheet(real_t(0));
    auto b = make_sheet(d);
    std::vector<decltype(tf::test::make_tagged_operand(a))> operands;
    operands.reserve(2);
    operands.push_back(tf::test::make_tagged_operand(a));
    operands.push_back(tf::test::make_tagged_operand(b));
    auto forms = tf::test::tagged_forms(operands);
    return tf::test::mesh_arrangements_of(
        tf::test::forms_range(forms), tf::intersect_config{mode, tolerance});
  };

  SECTION("gap inside the band welds the sheets") {
    auto [mesh, tag_labels, face_labels] = arrange(real_t(0.5e-3));
    REQUIRE(mesh.polygons().size() == 4u);
    REQUIRE(mesh.points().size() == 4u);
  }

  SECTION("gap beyond the band leaves two sheets") {
    auto [mesh, tag_labels, face_labels] = arrange(real_t(2e-3));
    REQUIRE(mesh.polygons().size() == 4u);
    REQUIRE(mesh.points().size() == 8u);
  }
}

TEST_CASE("mesh_arrangements_tolerance_box_plane",
          "[arrangements][tolerance]") {
  // Unit cube [-0.5, +0.5]^3 (volume = 1), with a planar quad at z = 0
  // whose corners sit a small `gap` INSIDE the cube walls.
  //
  // A TOLERANCE IS A STATEMENT ABOUT THE INPUT, NOT ABOUT A PREDICATE. It
  // moves each vertex at most `tol` onto a lattice point of the planes its
  // OWN faces state, and the arrangement of that moved mesh is then exact.
  // Nothing draws the quad's rim onto the cube's wall: the rim lies in one
  // plane it already stands on, and the cube's corners meet three planes of
  // their own. So the fin stays a fin at every tolerance, and the answer is
  // the exact answer of a mesh that moved by less than `tol`.
  using index_t = int;
  using real_t = float;

  const real_t gap = real_t(1e-4);
  const real_t half_extent = real_t(0.5) - gap;

  auto cube = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));

  tf::polygons_buffer<index_t, real_t, 3, 3> plane;
  plane.points_buffer().allocate(4);
  plane.faces_buffer().allocate(2);
  plane.points_buffer()[0] =
      tf::point<real_t, 3>{-half_extent, -half_extent, real_t(0)};
  plane.points_buffer()[1] =
      tf::point<real_t, 3>{+half_extent, -half_extent, real_t(0)};
  plane.points_buffer()[2] =
      tf::point<real_t, 3>{+half_extent, +half_extent, real_t(0)};
  plane.points_buffer()[3] =
      tf::point<real_t, 3>{-half_extent, +half_extent, real_t(0)};
  plane.faces_buffer()[0] = std::array<index_t, 3>{0, 1, 2};
  plane.faces_buffer()[1] = std::array<index_t, 3>{0, 2, 3};

  std::vector<decltype(tf::test::make_tagged_operand(cube))> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(cube));
  operands.push_back(tf::test::make_tagged_operand(plane));
  auto forms = tf::test::tagged_forms(operands);

  SECTION("tolerance >> gap: the fin is still a fin, the cube still whole") {
    const double tol = 1e-3; // tol >> gap (= 1e-4)

    auto [mesh, tag_labels, face_labels] = tf::test::mesh_arrangements_of(
        tf::test::forms_range(forms),
        tf::intersect_config{tf::intersect_mode::primitives, tol});

    tf::orient_faces_consistently(mesh.polygons());
    auto labels = tf::make_domain_labels(
        mesh.polygons(), tf::domain_config::ignore_open_fragments);

    // outer + cube interior; the quad is an open fragment either way
    REQUIRE(labels.n_domains == 2);

    auto [domain_meshes, domain_ids] =
        tf::split_into_domains(mesh.polygons(), labels);
    REQUIRE(domain_meshes.size() == 2);

    std::vector<double> volumes;
    for (auto &dm : domain_meshes)
      volumes.push_back(double(tf::signed_volume(dm.polygons())));
    std::sort(volumes.begin(), volumes.end());

    // the cube moved by less than the band, so its volume is the exact
    // volume of a cube whose walls each shifted by at most `tol`
    REQUIRE(volumes[0] == Catch::Approx(-1.0).margin(0.01));
    REQUIRE(volumes[1] == Catch::Approx(+1.0).margin(0.01));
  }

  SECTION("tolerance = 0: plane stays detached, cube remains whole") {
    auto [mesh, tag_labels, face_labels] = tf::test::mesh_arrangements_of(
        tf::test::forms_range(forms),
        tf::intersect_config{tf::intersect_mode::primitives, 0.0});

    tf::orient_faces_consistently(mesh.polygons());
    auto labels = tf::make_domain_labels(
        mesh.polygons(), tf::domain_config::ignore_open_fragments);

    // outer + cube interior (plane fin discarded by ignore_open_fragments)
    REQUIRE(labels.n_domains == 2);

    auto [domain_meshes, domain_ids] =
        tf::split_into_domains(mesh.polygons(), labels);
    REQUIRE(domain_meshes.size() == 2);

    std::vector<double> volumes;
    for (auto &dm : domain_meshes)
      volumes.push_back(double(tf::signed_volume(dm.polygons())));
    std::sort(volumes.begin(), volumes.end());

    REQUIRE(volumes[0] == Catch::Approx(-1.0).margin(0.01));
    REQUIRE(volumes[1] == Catch::Approx(+1.0).margin(0.01));
  }

  SECTION("return_source_ids drops the garbage class, stays aligned") {
    // tol = 0: the plane is a detached fin. ignore_open_fragments routes its
    // faces into the sentinel/garbage class, which `take(n_domains)` trims.
    // This exercises the offsets.reallocate(n_components + 1) trailing-block
    // drop in the source buffer.
    auto [mesh, tag_labels, face_labels] = tf::test::mesh_arrangements_of(
        tf::test::forms_range(forms),
        tf::intersect_config{tf::intersect_mode::primitives, 0.0});

    tf::orient_faces_consistently(mesh.polygons());
    auto labels = tf::make_domain_labels(
        mesh.polygons(), tf::domain_config::ignore_open_fragments);
    REQUIRE(labels.n_domains == 2);

    auto [domain_meshes, domain_ids, source] =
        tf::split_into_domains(mesh.polygons(), labels, tf::return_source_ids);

    // Garbage block is gone: source has exactly the kept domains, no empty
    // or stray trailing block.
    REQUIRE(domain_meshes.size() == 2);
    REQUIRE(source.size() == domain_meshes.size());
    REQUIRE(domain_ids.size() == domain_meshes.size());

    const auto n_faces = index_t(mesh.polygons().faces().size());
    for (std::size_t c = 0; c < domain_meshes.size(); ++c) {
      auto block = source[c];
      REQUIRE(std::size_t(block.size()) ==
              domain_meshes[c].polygons().faces().size());
      const auto dom = domain_ids[c];
      for (auto f : block) {
        REQUIRE(f >= index_t(0));
        REQUIRE(f < n_faces);
        // Bounds this domain, and is a cube face (tag 0) -- never a
        // garbage-collected plane-fin face (tag 1).
        auto sides = labels.labels[f];
        REQUIRE((sides[0] == dom || sides[1] == dom));
        REQUIRE(tag_labels[f] == 0);
      }
    }
  }
}

// A box divided by two coincident planes through the free path
// (mesh_arrangements -> cleaned -> domain_labels -> split): the stack
// yields exactly two closed half-domains, and keep-one emission follows
// orientation — same winding: the min tag carries both sides and the
// dead member serves none; opposite winding: each member serves exactly
// its outward side.
TEST_CASE("coincident plane stack divides a box (free path)",
          "[arrangements][stack]") {
  using Mesh = tf::polygons_buffer<int, double, 3, 3>;

  const bool reverse_second = GENERATE(false, true);
  const double tol = reverse_second ? 1e-6 : 0.0;

  auto plane1 = tf::make_plane_mesh(15.0, 15.0, 1, 1);
  auto plane2 = tf::make_plane_mesh(15.0, 15.0, 1, 1);
  if (reverse_second)
    tf::reverse_winding(plane2.faces());
  auto box = tf::make_box_mesh(10.0, 10.0, 10.0, 1, 1, 1);

  std::vector<Mesh> meshes{plane1, plane2, box};
  std::vector<decltype(tf::test::make_tagged_operand(meshes.front()))> operands;
  for (auto &m : meshes)
    operands.push_back(tf::test::make_tagged_operand(m));
  auto forms = tf::test::tagged_forms(operands);

  const auto icfg = tf::intersect_config{
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours |
          tf::intersect_mode::resolve_self_crossing_contours,
      1e-6};
  auto [arr, tag_labels, face_labels] =
      tf::test::mesh_arrangements_of(tf::test::forms_range(forms), icfg);
  (void)face_labels;

  auto [clean, face_map, point_map] = tf::cleaned(
      arr.polygons(), tf::clean_config(tol, false, true), tf::return_index_map);
  (void)point_map;
  const auto ctags = tf::reindexed(
      tf::make_range(tag_labels.data(), tag_labels.size()), face_map);

  auto labels = tf::make_domain_labels(
      clean.polygons(), tf::domain_config::ignore_open_fragments |
                            tf::domain_config::exclude_outer_shell);
  auto [subs, ids] = tf::split_into_domains(clean.polygons(), labels);
  (void)ids;

  REQUIRE(labels.n_domains == 2);
  REQUIRE(subs.size() == 2);
  for (auto &sub : subs) {
    CHECK(tf::is_closed(sub.polygons()));
    CHECK(std::abs(double(tf::signed_volume(sub.polygons())) - 500.0) < 1e-6);
  }

  const int sentinel = int(labels.n_domains);
  const int outer = int(labels.outer_shell_label);
  auto sides_of = [&](int tag) {
    std::pair<std::set<int>, std::set<int>> io;
    std::size_t f = 0;
    for (auto t : ctags) {
      if (t == tag) {
        const int in = labels.labels[f][0];
        const int out = labels.labels[f][1];
        if (in != sentinel && in != outer)
          io.first.insert(in);
        if (out != sentinel && out != outer)
          io.second.insert(out);
      }
      ++f;
    }
    return io;
  };
  auto [in0, out0] = sides_of(0);
  auto [in1, out1] = sides_of(1);
  if (!reverse_second) {
    CHECK((in0.size() == 1 && out0.size() == 1 && in0 != out0));
    CHECK((in1.empty() && out1.empty()));
  } else {
    CHECK((in0.empty() && in1.empty()));
    CHECK((out0.size() == 1 && out1.size() == 1 && out0 != out1));
  }
}

// A 3-deep coincident stack with mixed windings (up, down, up): the
// store aliases every dead member to the live survivor's triangles, and
// each member's winding flag must be relative to THAT survivor — not to
// a deeper dead member of the clique.
TEST_CASE("three-deep mixed-winding stack keeps consistent windings",
          "[arrangements][stack]") {
  auto plane1 = tf::make_plane_mesh(15.0, 15.0, 1, 1);
  auto plane2 = tf::make_plane_mesh(15.0, 15.0, 1, 1);
  auto plane3 = tf::make_plane_mesh(15.0, 15.0, 1, 1);
  tf::reverse_winding(plane2.faces());
  auto box = tf::make_box_mesh(10.0, 10.0, 10.0, 1, 1, 1);

  std::vector<tf::polygons_buffer<int, double, 3, 3>> meshes{plane1, plane2,
                                                             plane3, box};
  std::vector<decltype(tf::test::make_tagged_operand(meshes.front()))> operands;
  for (auto &m : meshes)
    operands.push_back(tf::test::make_tagged_operand(m));
  auto forms = tf::test::tagged_forms(operands);

  const auto icfg = tf::intersect_config{
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours |
          tf::intersect_mode::resolve_self_crossing_contours,
      1e-6};
  auto [arr, tag_labels, face_labels] =
      tf::test::mesh_arrangements_of(tf::test::forms_range(forms), icfg);
  (void)face_labels;

  // each member's emitted wall triangles must keep that member's own
  // winding: planes 1 and 3 face +z, plane 2 faces -z
  double zsum[3] = {0, 0, 0};
  auto pts = arr.polygons().points();
  for (std::size_t i = 0; i < arr.polygons().size(); ++i) {
    const int t = tag_labels[i];
    if (t > 2)
      continue;
    auto f = arr.polygons().faces()[i];
    auto n = tf::cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]]);
    zsum[t] += double(n[2]);
  }
  CHECK(zsum[0] > 0);
  CHECK(zsum[1] < 0);
  CHECK(zsum[2] > 0);
}

TEMPLATE_TEST_CASE("arrangement of a hexagon with a collinear leading run",
                   "[arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // THE ROTATION ORACLE: the arrangement of a polygon is a property of its
  // geometry, so which corner its cycle starts at cannot change the answer.
  auto blade = make_crossing_blade<index_t, real_t>();
  for (int shift = 0; shift < 6; ++shift) {
    INFO("corner cycle rotated by " << shift);
    auto hexagon = make_collinear_run_hexagon<index_t, real_t>(shift);
    auto hexagon_operand = tf::test::make_tagged_operand(hexagon);
    auto blade_operand = tf::test::make_tagged_operand(blade);
    auto graph = tf::test::build_pair_arrangement(hexagon_operand.form(),
                                                  blade_operand.form(), {});
    auto mesh = tf::test::arrangement_mesh_of(graph);
    CHECK(graph.failed().size() == 0);
    CHECK(mesh.polygons().size() == 16);
    CHECK(total_polygon_area(mesh) == Catch::Approx(108.0));
  }
}

TEST_CASE("refined graph materialises the refined arrangement mesh",
          "[arrangements][refined]") {
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto plane = tf::make_plane_mesh(4.0, 4.0, 1, 1);
  std::vector<decltype(tf::test::make_tagged_operand(box))> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(box));
  operands.push_back(tf::test::make_tagged_operand(plane));
  auto forms = tf::test::tagged_forms(operands);

  auto stock =
      tf::test::build_range_arrangement(tf::test::forms_range(forms), {});
  auto stock_mesh = tf::test::arrangement_mesh_of(stock);

  auto g = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::triangulation_type::refined_cdt);
  auto mesh = tf::test::arrangement_mesh_of(g);

  const int n_pts = int(mesh.points_buffer().size());
  REQUIRE(mesh.polygons().size() > 0);
  for (auto f : mesh.faces())
    for (int k = 0; k < 3; ++k) {
      CHECK(f[k] >= 0);
      CHECK(f[k] < n_pts);
    }
  // the refined path is live: quality insertion adds points and
  // triangles over the stock arrangement of the same forms
  CHECK(mesh.points_buffer().size() > stock_mesh.points_buffer().size());
  CHECK(mesh.polygons().size() > stock_mesh.polygons().size());
}

TEST_CASE("refined graph handles coplanar stacks (dead regions)",
          "[arrangements][refined]") {
  // the plane lies IN the box's top face: the shared wall is a
  // coplanar stack, so the refined pass runs with dead regions live
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto plane = tf::make_plane_mesh(4.0, 4.0, 1, 1);
  for (auto &&p : plane.points())
    p[2] = 1.0;
  std::vector<decltype(tf::test::make_tagged_operand(box))> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(box));
  operands.push_back(tf::test::make_tagged_operand(plane));
  auto forms = tf::test::tagged_forms(operands);

  auto stock =
      tf::test::build_range_arrangement(tf::test::forms_range(forms), {});
  REQUIRE(stock.coplanar_triples().size() > 0);
  auto stock_mesh = tf::test::arrangement_mesh_of(stock);

  auto g = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::triangulation_type::refined_cdt);
  auto mesh = tf::test::arrangement_mesh_of(g);

  REQUIRE(g.arrangement().failed().size() == 0);
  const int n_pts = int(mesh.points_buffer().size());
  REQUIRE(mesh.polygons().size() > 0);
  for (auto f : mesh.faces())
    for (int k = 0; k < 3; ++k) {
      CHECK(f[k] >= 0);
      CHECK(f[k] < n_pts);
    }
  CHECK(mesh.points_buffer().size() >= stock_mesh.points_buffer().size());
}

TEST_CASE("refined pooled carriers refine their overlap interiors",
          "[arrangements][refined]") {
  // two boxes whose coplanar top (and bottom) faces overlap in a skinny
  // strip: the strip is covered by BOTH members, and refinement must
  // treat it as interior — its slivers get quality insertions like any
  // single-covered region
  auto a = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto b = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  for (auto &&p : b.points())
    p[0] += 1.7;
  std::vector<decltype(tf::test::make_tagged_operand(a))> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(a));
  operands.push_back(tf::test::make_tagged_operand(b));
  auto forms = tf::test::tagged_forms(operands);

  auto g = tf::test::build_range_arrangement(
      tf::test::forms_range(forms), tf::triangulation_type::refined_cdt);
  REQUIRE(g.arrangement().failed().size() == 0);
  REQUIRE(g.coplanar_triples().size() > 0);

  const auto &created = g.created_points();
  const auto &conv = g.converter();
  const auto voffs = g.vertex_offsets();
  auto apply_form = g.apply_to_form();
  auto position = [&](int tag, const auto &v) {
    using pt_t = std::decay_t<decltype(created[0])>;
    if (v.source == tf::intersect::graph::vertex_source::created)
      return created[std::size_t(v.id)];
    pt_t out{};
    const auto local = v.id - voffs[std::size_t(tag)];
    apply_form(tag, [&](const auto &form) {
      out = conv.convert(form.points()[local]);
    });
    return out;
  };
  auto min_angle_cos = [&](const auto &p0, const auto &p1, const auto &p2) {
    // the LARGEST cosine over the three corners = the smallest angle
    double best = -1.0;
    const std::array<std::array<double, 3>, 3> q{
        {{double(p0[0]), double(p0[1]), double(p0[2])},
         {double(p1[0]), double(p1[1]), double(p1[2])},
         {double(p2[0]), double(p2[1]), double(p2[2])}}};
    for (int c = 0; c < 3; ++c) {
      const auto &o = q[std::size_t(c)];
      const auto &u = q[std::size_t((c + 1) % 3)];
      const auto &w = q[std::size_t((c + 2) % 3)];
      double du[3], dw[3], nu = 0, nw = 0, dot = 0;
      for (int k = 0; k < 3; ++k) {
        du[k] = u[std::size_t(k)] - o[std::size_t(k)];
        dw[k] = w[std::size_t(k)] - o[std::size_t(k)];
        nu += du[k] * du[k];
        nw += dw[k] * dw[k];
        dot += du[k] * dw[k];
      }
      if (nu == 0.0 || nw == 0.0)
        continue;
      best = std::max(best, dot / std::sqrt(nu * nw));
    }
    return best;
  };

  auto tris = g.global().exposed_tris();
  auto stacked = g.stacked();
  auto tags = g.triangle_tags();
  auto dead = g.dead();
  std::size_t bad = 0, total = 0;
  const double cos15 = std::cos(15.0 * 3.14159265358979323846 / 180.0);
  for (std::size_t t = 0; t < tris.size(); ++t) {
    if (!stacked[t] || dead[t])
      continue;
    const int tag = int(tags[t]);
    const auto &tv = tris[t];
    ++total;
    if (min_angle_cos(position(tag, tv[0]), position(tag, tv[1]),
                      position(tag, tv[2])) > cos15)
      ++bad;
  }
  REQUIRE(total > 0);
  // the overlap strip must be refined: no sliver population survives
  CHECK(bad == 0);
}
