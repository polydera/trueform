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
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <set>
#include <trueform/trueform.hpp>
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

template <typename Mesh> auto count_degenerate(const Mesh &mesh) -> int {
  int n = 0;
  for (auto face : mesh.faces())
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
      ++n;
  return n;
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

  auto p0 = s0.polygons() | tf::tag(f0);
  auto p1 = s1.polygons() | tf::tag(f1);
  auto p2 = s2.polygons() | tf::tag(f2);
  decltype(p0) forms[] = {p0, p1, p2};

  auto [mesh, tag_labels, face_labels, curves] = tf::make_mesh_arrangements(
      tf::make_range(forms, forms + 3), tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);

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

  auto [mesh, tag_labels, face_labels, curves] = tf::make_mesh_arrangements(
      s0.polygons(), s1.polygons() | tf::tag(frame), tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);

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

  const std::array forms{
      box.polygons(),
      cylinder.polygons(),
      sphere.polygons(),
  };

  auto [mesh, tag_labels, face_labels, curves] = tf::make_mesh_arrangements(
      tf::make_range(forms.begin(), forms.end()),
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours,
      tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);

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
    const std::array forms{a.polygons(), b.polygons()};
    return tf::make_mesh_arrangements(
        tf::make_range(forms.begin(), forms.end()),
        tf::intersect_config{mode, tolerance});
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
  // whose corners sit a small `gap` INSIDE the cube walls. With sufficient
  // tolerance, the plane snaps onto the cube faces and splits the cube
  // into two halves of volume 0.5 each. With zero tolerance, the plane
  // stays a detached fin and the cube remains a single closed region.
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

  const std::array forms{cube.polygons(), plane.polygons()};

  SECTION("tolerance >> gap: plane snaps to cube, splits into 2 halves") {
    const double tol = 1e-3; // tol >> gap (= 1e-4)

    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        tf::make_range(forms.begin(), forms.end()),
        tf::intersect_config{tf::intersect_mode::primitives, tol});

    auto cleaned = tf::cleaned(mesh.polygons(), real_t(tol));
    tf::orient_faces_consistently(cleaned.polygons());
    auto labels = tf::make_domain_labels(
        cleaned.polygons(), tf::domain_config::ignore_open_fragments);

    // outer + upper half + lower half
    REQUIRE(labels.n_domains == 3);

    auto [domain_meshes, domain_ids] =
        tf::split_into_domains(cleaned.polygons(), labels);
    REQUIRE(domain_meshes.size() == 3);

    std::vector<double> volumes;
    for (auto &dm : domain_meshes)
      volumes.push_back(double(tf::signed_volume(dm.polygons())));
    std::sort(volumes.begin(), volumes.end());

    // outer (cube surface with inward normals) → -1; two inner halves → +0.5
    // each
    REQUIRE(volumes[0] == Catch::Approx(-1.0).margin(0.01));
    REQUIRE(volumes[1] == Catch::Approx(+0.5).margin(0.01));
    REQUIRE(volumes[2] == Catch::Approx(+0.5).margin(0.01));
  }

  SECTION("tolerance = 0: plane stays detached, cube remains whole") {
    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        tf::make_range(forms.begin(), forms.end()),
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
    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        tf::make_range(forms.begin(), forms.end()),
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

TEST_CASE("split_into_domains return_source_ids (box provenance)",
          "[arrangements][domains][source_ids]") {
  using index_t = int;
  using real_t = float;

  auto box = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::orient_faces_consistently(box.polygons());

  auto labels = tf::make_domain_labels(box.polygons());
  // A closed box bounds two domains: the interior and the outer shell.
  REQUIRE(labels.n_domains == 2);

  auto [meshes, domain_ids, source] =
      tf::split_into_domains(box.polygons(), labels, tf::return_source_ids);

  // Source blocks run parallel to the meshes; no empty trailing block.
  REQUIRE(source.size() == meshes.size());
  REQUIRE(domain_ids.size() == meshes.size());

  const auto n_faces = index_t(box.polygons().faces().size());
  for (std::size_t c = 0; c < meshes.size(); ++c) {
    auto block = source[c];
    // One source face id per emitted face, in emitted-face order.
    REQUIRE(std::size_t(block.size()) == meshes[c].polygons().faces().size());
    const auto dom = domain_ids[c];
    for (auto f : block) {
      REQUIRE(f >= index_t(0));
      REQUIRE(f < n_faces);
      // Original face f bounds this domain on at least one of its two sides.
      auto sides = labels.labels[f];
      REQUIRE((sides[0] == dom || sides[1] == dom));
    }
  }

  // The plain overload is unchanged and agrees on meshes/labels.
  auto [meshes2, domain_ids2] = tf::split_into_domains(box.polygons(), labels);
  REQUIRE(meshes2.size() == meshes.size());
  REQUIRE(domain_ids2.size() == domain_ids.size());
}

TEST_CASE("split_into_domains return_source_ids (empty input)",
          "[arrangements][domains][source_ids]") {
  // No faces: offsets is empty, so the source-buffer trim must not reallocate
  // to size 1 and read an uninitialized offset. Everything comes back empty.
  using index_t = int;
  using real_t = float;
  tf::polygons_buffer<index_t, real_t, 3, 3> empty;
  auto labels = tf::make_domain_labels(empty.polygons());

  auto [meshes, domain_ids, source] =
      tf::split_into_domains(empty.polygons(), labels, tf::return_source_ids);
  REQUIRE(meshes.size() == 0);
  REQUIRE(domain_ids.size() == 0);
  REQUIRE(source.size() == 0);
}

// ---------------------------------------------------------------------------
// outer_shell_label correctness. The outer shell is the domain with the
// most-negative signed volume (tf::signed_volume). Two disjoint spheres, each
// cut by two intersecting planes: two distinct closed bodies whose exteriors
// merge into one outer shell, plus open plane fins outside the spheres --- so
// every domain_config combination exercises both the merge and both flags.
// ---------------------------------------------------------------------------

TEST_CASE("domain_outer_shell_two_spheres_two_planes",
          "[arrangements][domains]") {
  using index_t = int;
  using real_t = double;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(1), 24, 24);
  auto &pts1 = s1.points_buffer();
  for (std::size_t i = 0; i < pts1.size(); ++i)
    pts1[i][0] += real_t(4); // s1 clear of s0

  // Two intersecting planes (z = 0 and y = 0), spanning both spheres and
  // extending past them so their parts outside the spheres are open fins.
  auto plane_z = tf::make_plane_mesh<index_t>(real_t(12), real_t(12)); // z = 0
  tf::polygons_buffer<index_t, real_t, 3, 3> plane_y;                  // y = 0
  plane_y.points_buffer().allocate(4);
  plane_y.faces_buffer().allocate(2);
  plane_y.points_buffer()[0] =
      tf::point<real_t, 3>{real_t(-6), real_t(0), real_t(-6)};
  plane_y.points_buffer()[1] =
      tf::point<real_t, 3>{real_t(6), real_t(0), real_t(-6)};
  plane_y.points_buffer()[2] =
      tf::point<real_t, 3>{real_t(6), real_t(0), real_t(6)};
  plane_y.points_buffer()[3] =
      tf::point<real_t, 3>{real_t(-6), real_t(0), real_t(6)};
  plane_y.faces_buffer()[0] = std::array<index_t, 3>{0, 1, 2};
  plane_y.faces_buffer()[1] = std::array<index_t, 3>{0, 2, 3};

  const std::array forms{s0.polygons(), s1.polygons(), plane_z.polygons(),
                         plane_y.polygons()};
  auto [mesh, tag_labels, face_labels] =
      tf::make_mesh_arrangements(tf::make_range(forms.begin(), forms.end()));
  auto cleaned = tf::cleaned(mesh.polygons(), real_t(1e-7));
  tf::orient_faces_consistently(cleaned.polygons());

  auto check = [&](tf::domain_config config) {
    auto labels = tf::make_domain_labels(cleaned.polygons(), config);
    const bool exclude = (config & tf::domain_config::exclude_outer_shell);

    if (exclude) {
      // Universe folded to the sentinel one past the valid range; some
      // face-side carries it.
      REQUIRE(labels.outer_shell_label == labels.n_domains);
      bool carried = false;
      for (auto v : labels.labels.data_buffer())
        if (index_t(v) == index_t(labels.n_domains)) {
          carried = true;
          break;
        }
      REQUIRE(carried);
    } else {
      // Outer shell is a real domain: the most-negative signed volume, and
      // the two bodies' exteriors merged into exactly one negative domain.
      REQUIRE(labels.outer_shell_label >= 0);
      REQUIRE(labels.outer_shell_label < labels.n_domains);

      auto [domain_meshes, domain_ids] =
          tf::split_into_domains(cleaned.polygons(), labels);
      REQUIRE(domain_meshes.size() > 0);

      int n_negative = 0;
      double min_vol = double(tf::signed_volume(domain_meshes[0].polygons()));
      index_t outer = index_t(domain_ids[0]);
      for (std::size_t i = 0; i < domain_meshes.size(); ++i) {
        double v = double(tf::signed_volume(domain_meshes[i].polygons()));
        if (v < 0.0)
          ++n_negative;
        if (v < min_vol) {
          min_vol = v;
          outer = index_t(domain_ids[i]);
        }
      }
      REQUIRE(n_negative == 1);
      REQUIRE(outer == index_t(labels.outer_shell_label));
    }
  };

  check(tf::domain_config::none);
  check(tf::domain_config::ignore_open_fragments);
  check(tf::domain_config::exclude_outer_shell);
  check(tf::domain_config::exclude_outer_shell |
        tf::domain_config::ignore_open_fragments);
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
  std::vector<decltype(meshes.front().polygons())> forms;
  for (auto &m : meshes)
    forms.push_back(m.polygons());

  const auto icfg = tf::intersect_config{
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours |
          tf::intersect_mode::resolve_self_crossing_contours,
      1e-6};
  auto [arr, tag_labels, face_labels] = tf::make_mesh_arrangements(
      tf::make_range(forms.data(), forms.data() + forms.size()), icfg);
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
  std::vector<decltype(meshes.front().polygons())> forms;
  for (auto &m : meshes)
    forms.push_back(m.polygons());

  const auto icfg = tf::intersect_config{
      tf::intersect_mode::primitives |
          tf::intersect_mode::resolve_crossing_contours |
          tf::intersect_mode::resolve_self_crossing_contours,
      1e-6};
  auto [arr, tag_labels, face_labels] = tf::make_mesh_arrangements(
      tf::make_range(forms.data(), forms.data() + forms.size()), icfg);
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

TEST_CASE("refined graph materialises the refined arrangement mesh",
          "[arrangements][refined]") {
  auto box = tf::make_box_mesh(2.0, 2.0, 2.0, 1, 1, 1);
  auto plane = tf::make_plane_mesh(4.0, 4.0, 1, 1);
  std::vector<decltype(box.polygons())> forms{box.polygons(),
                                              plane.polygons()};

  auto stock = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()));
  auto stock_mesh = tf::make_arrangement_mesh(stock);

  auto g = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()),
      tf::triangulation_type::refined_cdt);
  auto mesh = tf::make_arrangement_mesh(g);

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
  std::vector<decltype(box.polygons())> forms{box.polygons(),
                                              plane.polygons()};

  auto stock = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()));
  REQUIRE(stock.dead_loops().size() > 0);
  auto stock_mesh = tf::make_arrangement_mesh(stock);

  auto g = tf::make_arrangement_graph(
      tf::make_range(forms.data(), forms.data() + forms.size()),
      tf::triangulation_type::refined_cdt);
  auto mesh = tf::make_arrangement_mesh(g);

  REQUIRE(g.triangulations().n_failed == 0);
  const int n_pts = int(mesh.points_buffer().size());
  REQUIRE(mesh.polygons().size() > 0);
  for (auto f : mesh.faces())
    for (int k = 0; k < 3; ++k) {
      CHECK(f[k] >= 0);
      CHECK(f[k] < n_pts);
    }
  CHECK(mesh.points_buffer().size() >= stock_mesh.points_buffer().size());
}
