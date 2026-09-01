/**
 * @file test_domain_labels.cpp
 * @brief tf::make_domain_labels — stacked duplicate faces resolve like
 * the csg path's keep-one, and the outer shell is the domain the
 * unbounded universe occupies.
 *
 * A coincident duplicate stack (same vertex-id cycle, either winding)
 * keeps one copy; dead copies carry the sentinel pair and no extraction
 * emits them. The primary fixture is the arrangement of two boxes
 * sharing a wall: the emitted mesh carries both conformed wall fans
 * with shared vertex ids, and the labeler must resolve them to two unit
 * cells with no zero-volume pillow between the copies — matching
 * tf::make_csg_domains on the same input.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>

#include <array>
#include <cstddef>

TEST_CASE("domain labels: arrangement stacks resolve to two cells",
          "[topology][domains][stacks]") {
  using index_t = std::int32_t;
  using real_t = double;
  auto a = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  auto b = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::ensure_positive_orientation(a.polygons());
  tf::ensure_positive_orientation(b.polygons());
  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{1.0, 0.0, 0.0}));
  auto soup = tf::concatenated(a.polygons(), b.polygons() | tf::tag(frame));
  auto [arranged, labels] = tf::make_polygon_arrangements(
      soup.polygons(),
      {tf::intersect_mode::primitives | tf::intersect_mode::resolve_contours});
  (void)labels;

  // the arrangement emits both conformed wall fans (a stack of
  // id-identical, opposite-wound triangles) — no clean step
  auto dl = tf::make_domain_labels(arranged.polygons(),
                                   tf::domain_config::exclude_outer_shell);
  auto [cells, ids] = tf::split_into_domains(arranged.polygons(), dl);

  REQUIRE(ids.size() == 2);
  for (auto &cell : cells) {
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(tf::is_manifold(cell.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(cell.polygons())),
                 Catch::Matchers::WithinRel(1.0, 1e-9));
  }
}

namespace {

// two unit boxes sharing the wall at x=0.5; the wall stored TWICE with
// opposite windings and identical cycles. Faces 0-19 are the hulls
// (10 quads x 2 tris), 20/21 the +x-wound wall (out of the left box),
// 22/23 the -x-wound duplicate (out of the right box).
auto make_two_box_stack() {
  tf::polygons_buffer<std::int32_t, double, 3, 3> m;
  auto &pts = m.points_buffer();
  pts.emplace_back(-0.5, -0.5, -0.5); // 0
  pts.emplace_back(-0.5, 0.5, -0.5);  // 1
  pts.emplace_back(-0.5, 0.5, 0.5);   // 2
  pts.emplace_back(-0.5, -0.5, 0.5);  // 3
  pts.emplace_back(0.5, -0.5, -0.5);  // 4
  pts.emplace_back(0.5, 0.5, -0.5);   // 5
  pts.emplace_back(0.5, 0.5, 0.5);    // 6
  pts.emplace_back(0.5, -0.5, 0.5);   // 7
  pts.emplace_back(1.5, -0.5, -0.5);  // 8
  pts.emplace_back(1.5, 0.5, -0.5);   // 9
  pts.emplace_back(1.5, 0.5, 0.5);    // 10
  pts.emplace_back(1.5, -0.5, 0.5);   // 11
  auto &f = m.faces_buffer();
  auto quad = [&](int p, int q, int r, int s) {
    f.emplace_back(p, q, r);
    f.emplace_back(p, r, s);
  };
  quad(0, 3, 2, 1);
  quad(0, 1, 5, 4);
  quad(1, 2, 6, 5);
  quad(2, 3, 7, 6);
  quad(3, 0, 4, 7);
  quad(8, 9, 10, 11);
  quad(4, 5, 9, 8);
  quad(5, 6, 10, 9);
  quad(6, 7, 11, 10);
  quad(7, 4, 8, 11);
  quad(4, 5, 6, 7); // wall, wound +x
  quad(4, 7, 6, 5); // wall duplicate, wound -x
  return m;
}

} // namespace

TEST_CASE("domain labels: retained duplicate wall emits no pillow cell",
          "[topology][domains][stacks]") {
  auto m = make_two_box_stack();

  auto dl = tf::make_domain_labels(m.polygons(),
                                   tf::domain_config::exclude_outer_shell);
  auto [cells, ids] = tf::split_into_domains(m.polygons(), dl);

  // keep-one: the wall bounds the two boxes once; no zero-volume
  // pillow between the copies
  REQUIRE(ids.size() == 2);
  for (auto &cell : cells) {
    REQUIRE(tf::is_closed(cell.polygons()));
    REQUIRE(tf::is_manifold(cell.polygons()));
    REQUIRE_THAT(static_cast<double>(tf::signed_volume(cell.polygons())),
                 Catch::Matchers::WithinRel(1.0, 1e-9));
  }
}

TEST_CASE("make_domain_labels: reversed coincident wall stack seals",
          "[domains][stacks]") {
  // A box cut by a sheet and its winding-reversed coincident copy,
  // through the mesh path with duplicate faces kept: the stack
  // resolution pairs the identical twins and the wall seals into two
  // closed halves.
  using index_t = std::int32_t;
  using real_t = double;
  using mesh3_t = tf::polygons_buffer<index_t, real_t, 3, 3>;
  mesh3_t box = tf::triangulated(
      tf::make_box_mesh<index_t, real_t>(real_t(2), real_t(2), real_t(2))
          .polygons());
  tf::ensure_positive_orientation(box.polygons());
  mesh3_t plane = tf::triangulated(
      tf::make_plane_mesh<index_t, real_t>(real_t(4), real_t(4)).polygons());
  mesh3_t plane_rev = plane;
  for (auto &&face : plane_rev.faces_buffer())
    std::swap(face[0], face[2]);

  std::vector<decltype(box.polygons())> forms{
      box.polygons(), plane.polygons(), plane_rev.polygons()};
  auto arr = tf::make_mesh_arrangements(tf::make_range(forms));
  auto clean = tf::cleaned(std::get<0>(arr).polygons(),
                           tf::clean_config(real_t(1e-9), false, true));
  auto dl = tf::make_domain_labels(
      clean.polygons(), tf::domain_config::exclude_outer_shell |
                            tf::domain_config::ignore_open_fragments);
  auto [cells, ids] = tf::split_into_domains(clean.polygons(), dl);
  REQUIRE(cells.size() == 2);
  for (auto &c : cells) {
    REQUIRE(tf::is_closed(c.polygons()));
    REQUIRE(tf::is_manifold(c.polygons()));
    REQUIRE_THAT(std::abs(double(tf::signed_volume(c.polygons()))),
                 Catch::Matchers::WithinAbs(4.0, 1e-9));
  }
}

TEST_CASE("domain labels: opposing stack members serve their own side",
          "[topology][domains][stacks][provenance]") {
  using index_t = std::int32_t;
  auto m = make_two_box_stack();

  auto dl = tf::make_domain_labels(m.polygons(),
                                   tf::domain_config::exclude_outer_shell);
  auto [cells, ids, src] =
      tf::split_into_domains(m.polygons(), dl, tf::return_source_ids);
  REQUIRE(cells.size() == 2);

  // each wall face is emitted exactly once, into the box it is wound
  // out of: +x copy (faces 20, 21) left, -x copy (22, 23) right
  for (std::size_t k = 0; k < cells.size(); ++k) {
    auto cpts = cells[k].polygons().points();
    double cx = 0;
    for (auto p : cpts)
      cx += double(p[0]);
    cx /= double(cpts.size());
    const bool left = cx < 0.5;
    int n_wall = 0;
    for (std::size_t j = 0; j < cells[k].faces().size(); ++j) {
      const index_t s = src[k][j];
      if (s < 20)
        continue;
      ++n_wall;
      REQUIRE((left ? s < 22 : s >= 22));
    }
    REQUIRE(n_wall == 2);
  }
}

// ---------------------------------------------------------------------------
// outer_shell_label correctness. The outer shell is the domain with the
// most-negative signed volume (tf::signed_volume). Two disjoint spheres, each
// cut by two intersecting planes: two distinct closed bodies whose exteriors
// merge into one outer shell, plus open plane fins outside the spheres --- so
// every domain_config combination exercises both the merge and both flags.
// ---------------------------------------------------------------------------

TEST_CASE("domain_outer_shell_two_spheres_two_planes",
          "[topology][domains]") {
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
