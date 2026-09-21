/**
 * @file test_self_arrangements.cpp
 * @brief Tests for self-intersection: make_polygon_arrangements produces
 *        the same results as make_mesh_arrangements on equivalent geometry,
 *        and cuts a coplanar group along exactly the overlap boundary.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <trueform/trueform.hpp>
#include <vector>

#include "arrangement_builders.hpp"
#include "arrangement_readers.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"

template <typename Paths, typename index_t>
auto count_endpoints(const Paths &paths) -> int {
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
auto self_arrangements_count_degenerate(const Mesh &mesh) -> int {
  int n = 0;
  for (auto face : mesh.faces())
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
      ++n;
  return n;
}

TEMPLATE_TEST_CASE("Self arrangements: 3 spheres match mesh arrangements",
                   "[self_arrangements]",
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
  auto merged = tf::concatenated(tf::test::forms_range(forms));

  // Reference: make_mesh_arrangements
  auto [ref_mesh, ref_tags, ref_faces, ref_curves] =
      tf::test::mesh_arrangements_with_curves_of(tf::test::forms_range(forms),
                                                 tf::intersect_config{});

  SECTION("make_polygon_arrangements") {
    auto merged_operand = tf::test::make_tagged_operand(merged);
    auto [mesh, face_labels, curves] =
        tf::test::polygon_arrangements_with_curves_of(
            merged_operand.form(), tf::intersect_config{});

    REQUIRE(self_arrangements_count_degenerate(mesh) == 0);
    REQUIRE(mesh.faces().size() == ref_mesh.faces().size());
    REQUIRE(mesh.points().size() == ref_mesh.points().size());
    REQUIRE(face_labels.size() == mesh.faces().size());
    REQUIRE(curves.paths().size() == ref_curves.paths().size());
    REQUIRE(count_endpoints<decltype(curves.paths()), index_t>(curves.paths()) ==
            count_endpoints<decltype(ref_curves.paths()), index_t>(
                ref_curves.paths()));
  }
}

TEST_CASE("Coplanar group with a rewritten transversal copy stays exact",
          "[self_arrangements][coplanar]") {
  // A is coplanar with B (stamped records); C folds out of plane and
  // shares an edge with B that lies inside A. The (A, C) records are
  // rewritten across the shared edge into the flagged (A, B) group with
  // their flags cleared — the any-of coplanar routing must still cut A
  // along exactly the overlap boundary: every arrangement edge interior
  // to A lies on B's boundary.
  using index_t = int;
  tf::polygons_buffer<index_t, float, 3, 3> mesh;
  auto &pts = mesh.points_buffer();
  pts.allocate(7);
  pts[0] = tf::point<float, 3>{0.f, 0.f, 0.f};
  pts[1] = tf::point<float, 3>{6.f, 0.f, 0.f};
  pts[2] = tf::point<float, 3>{0.f, 6.f, 0.f};
  pts[3] = tf::point<float, 3>{1.f, 1.f, 0.f};
  pts[4] = tf::point<float, 3>{3.f, 1.f, 0.f};
  pts[5] = tf::point<float, 3>{1.f, 3.f, 0.f};
  pts[6] = tf::point<float, 3>{2.f, 1.f, -2.f};
  mesh.faces_buffer().emplace_back(0, 1, 2); // A
  mesh.faces_buffer().emplace_back(3, 4, 5); // B, inside A, coplanar
  mesh.faces_buffer().emplace_back(4, 3, 6); // C, folded down off B's edge

  auto mesh_operand = tf::test::make_tagged_operand(mesh);
  auto [arranged, labels] = tf::test::polygon_arrangements_of(
      mesh_operand.form(), tf::intersect_config{});

  auto close = [](float a, float b) { return std::abs(a - b) < 1e-4f; };
  auto on_a_boundary = [&](const auto &p) {
    return close(p[1], 0.f) || close(p[0], 0.f) || close(p[0] + p[1], 6.f);
  };
  auto on_b_boundary = [&](const auto &p) {
    bool e0 = close(p[1], 1.f) && p[0] > 1.f - 1e-4f && p[0] < 3.f + 1e-4f;
    bool e1 = close(p[0], 1.f) && p[1] > 1.f - 1e-4f && p[1] < 3.f + 1e-4f;
    bool e2 = close(p[0] + p[1], 4.f) && p[0] > 1.f - 1e-4f &&
              p[1] > 1.f - 1e-4f;
    return e0 || e1 || e2;
  };

  auto apts = arranged.polygons().points();
  double area_a = 0;
  for (std::size_t f = 0; f < arranged.polygons().size(); ++f) {
    if (labels[f] != 0) // descendants of A only
      continue;
    auto face = arranged.polygons().faces()[f];
    // accumulate area of A's pieces
    auto p0 = apts[face[0]];
    auto p1 = apts[face[1]];
    auto p2 = apts[face[2]];
    area_a += 0.5 * std::abs(double(p1[0] - p0[0]) * double(p2[1] - p0[1]) -
                             double(p1[1] - p0[1]) * double(p2[0] - p0[0]));
    // every edge of an A piece with BOTH endpoints off A's boundary
    // must lie on B's boundary — no stray chords from the mixed group
    for (int e = 0; e < 3; ++e) {
      auto q0 = apts[face[e]];
      auto q1 = apts[face[(e + 1) % 3]];
      auto mid = tf::point<float, 3>{(q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2,
                                     (q0[2] + q1[2]) / 2};
      if (!on_a_boundary(q0) && !on_a_boundary(q1)) {
        REQUIRE(on_b_boundary(q0));
        REQUIRE(on_b_boundary(q1));
        REQUIRE(on_b_boundary(mid));
      }
    }
  }
  // A's pieces tile A exactly
  REQUIRE_THAT(area_a, Catch::Matchers::WithinAbs(18.0, 1e-3));
}

/// The form that meets itself: a wide sheet at y = 0 pierced by a narrow
/// one at x = 0, both in the same mesh, so their crossing is stated by a
/// SELF record and the two pierce points are carried by the narrow
/// sheet's own edges.
template <typename Index, typename Real>
auto self_flat_crossing_sheets() -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  auto &pts = mesh.points_buffer();
  pts.allocate(8);
  pts[0] = tf::point<Real, 3>{Real(-2), Real(0), Real(-2)};
  pts[1] = tf::point<Real, 3>{Real(2), Real(0), Real(-2)};
  pts[2] = tf::point<Real, 3>{Real(2), Real(0), Real(2)};
  pts[3] = tf::point<Real, 3>{Real(-2), Real(0), Real(2)};
  pts[4] = tf::point<Real, 3>{Real(0), Real(-1), Real(-1)};
  pts[5] = tf::point<Real, 3>{Real(0), Real(1), Real(-1)};
  pts[6] = tf::point<Real, 3>{Real(0), Real(1), Real(1)};
  pts[7] = tf::point<Real, 3>{Real(0), Real(-1), Real(1)};
  mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
  mesh.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
  mesh.faces_buffer().emplace_back(Index(4), Index(5), Index(6));
  mesh.faces_buffer().emplace_back(Index(4), Index(6), Index(7));
  return mesh;
}

/// A box that meets neither sheet and stands inside their bounding box,
/// so it gives the sheets a nonzero vertex offset without moving the
/// lattice the two builds are stated on.
template <typename Index, typename Real>
auto self_flat_disjoint_box() -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  auto &pts = mesh.points_buffer();
  pts.allocate(8);
  const Real lo_x(1.1), hi_x(1.3), lo_y(0.4), hi_y(0.6), lo_z(1.1), hi_z(1.3);
  pts[0] = tf::point<Real, 3>{lo_x, lo_y, lo_z};
  pts[1] = tf::point<Real, 3>{hi_x, lo_y, lo_z};
  pts[2] = tf::point<Real, 3>{hi_x, hi_y, lo_z};
  pts[3] = tf::point<Real, 3>{lo_x, hi_y, lo_z};
  pts[4] = tf::point<Real, 3>{lo_x, lo_y, hi_z};
  pts[5] = tf::point<Real, 3>{hi_x, lo_y, hi_z};
  pts[6] = tf::point<Real, 3>{hi_x, hi_y, hi_z};
  pts[7] = tf::point<Real, 3>{lo_x, hi_y, hi_z};
  const Index corners[12][3] = {{0, 3, 2}, {0, 2, 1}, {4, 5, 6}, {4, 6, 7},
                                {0, 1, 5}, {0, 5, 4}, {3, 7, 6}, {3, 6, 2},
                                {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
  for (const auto &face : corners)
    mesh.faces_buffer().emplace_back(face[0], face[1], face[2]);
  return mesh;
}

/// One operand's emitted faces as coordinate triples, each rotated onto
/// its smallest corner and the list sorted — the arrangement's product,
/// read without its naming or its order.
template <typename Mesh, typename Labels>
auto self_flat_product_of_tag(const Mesh &mesh, const Labels &tags, int tag)
    -> std::vector<std::array<double, 9>> {
  std::vector<std::array<double, 9>> out;
  auto points = mesh.polygons().points();
  auto faces = mesh.polygons().faces();
  for (std::size_t f = 0; f < faces.size(); ++f) {
    if (int(tags[f]) != tag)
      continue;
    std::array<std::array<double, 3>, 3> corner;
    for (std::size_t k = 0; k < 3; ++k) {
      auto p = points[std::size_t(faces[f][k])];
      corner[k] = {double(p[0]), double(p[1]), double(p[2])};
    }
    std::size_t first = 0;
    for (std::size_t k = 1; k < 3; ++k)
      if (corner[k] < corner[first])
        first = k;
    std::array<double, 9> row{};
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t c = 0; c < 3; ++c)
        row[3 * k + c] = corner[(first + k) % 3][c];
    out.push_back(row);
  }
  std::sort(out.begin(), out.end());
  return out;
}

TEMPLATE_TEST_CASE("Self arrangements: a disjoint operand does not move a "
                   "form's self cut",
                   "[self_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  // A self record names its carrier edge in the FLAT vertex space, so the
  // form's own arrangement cannot depend on the offset its position in the
  // operand list gives it. The disjoint box states nothing about the
  // sheets and stands inside their box, so both builds run on one lattice
  // and the sheets' product must come back identical.
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  const auto config = tf::arrangement_config{tf::intersect_config{
      tf::intersect_mode::primitives | tf::intersect_mode::within, 0.0}};

  std::vector<tf::test::tagged_operand<index_t, real_t, 3>> alone;
  alone.reserve(1);
  alone.push_back(tf::test::make_tagged_operand(
      self_flat_crossing_sheets<index_t, real_t>()));
  auto alone_forms = tf::test::tagged_forms(alone);

  std::vector<tf::test::tagged_operand<index_t, real_t, 3>> offset;
  offset.reserve(2);
  offset.push_back(
      tf::test::make_tagged_operand(self_flat_disjoint_box<index_t, real_t>()));
  offset.push_back(tf::test::make_tagged_operand(
      self_flat_crossing_sheets<index_t, real_t>()));
  auto offset_forms = tf::test::tagged_forms(offset);

  auto alone_graph =
      tf::test::build_range_arrangement(tf::test::forms_range(alone_forms),
                                        config);
  auto offset_graph =
      tf::test::build_range_arrangement(tf::test::forms_range(offset_forms),
                                        config);
  REQUIRE(alone_graph.failed().size() == std::size_t(0));
  REQUIRE(offset_graph.failed().size() == std::size_t(0));

  auto [alone_mesh, alone_tags, alone_faces] =
      tf::test::mesh_arrangements_of(tf::test::forms_range(alone_forms),
                                     config);
  auto [offset_mesh, offset_tags, offset_faces] =
      tf::test::mesh_arrangements_of(tf::test::forms_range(offset_forms),
                                     config);

  const auto alone_product =
      self_flat_product_of_tag(alone_mesh, alone_tags, 0);
  const auto offset_product =
      self_flat_product_of_tag(offset_mesh, offset_tags, 1);
  REQUIRE(alone_product.size() > std::size_t(4));
  REQUIRE(alone_product == offset_product);
}
