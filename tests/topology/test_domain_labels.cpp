/**
 * @file test_domain_labels.cpp
 * @brief tf::make_domain_labels — stacked duplicate faces resolve like
 * the csg path's keep-one.
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

TEST_CASE("domain labels: retained duplicate wall emits no pillow cell",
          "[topology][domains][stacks]") {
  using index_t = std::int32_t;
  using real_t = double;
  // two unit boxes sharing the wall at x=0.5; the wall stored TWICE
  // with opposite windings and identical cycles
  tf::polygons_buffer<index_t, real_t, 3, 3> m;
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
  quad(4, 7, 6, 5); // wall duplicate, wound -x, same cycles reversed

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
