/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include "parity_checks.hpp"

#include "trueform/cpp/arrangement/mesh_arrangements.hpp"
#include "trueform/cpp/arrangement/polygon_arrangements.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"

#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace tf::cpp::test;

namespace {

/// The views a range entry takes, over operands the suite keeps.
template <typename Real>
auto parity_meshes_of(const std::vector<parity_operand<Real>> &operands)
    -> std::vector<typename parity_operand<Real>::mesh_type> {
  std::vector<typename parity_operand<Real>::mesh_type> meshes;
  meshes.reserve(operands.size());
  for (const auto &operand : operands)
    meshes.push_back(operand.mesh());
  return meshes;
}

/// A result read as an operand: the storage is core's, and the cache and the
/// identity placement are what a reading of it needs.
template <typename Real>
auto parity_operand_of(parity_mesh<Real> polygons) -> parity_operand<Real> {
  parity_operand<Real> result;
  result.polygons = std::move(polygons);
  return result;
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python-parity arrangement curve producers agree canonically",
    "[cpp][arrangement][python-parity][arrangements]", float, double) {
  const std::vector<parity_operand<TestType>> inputs{
      horizontal_triangle<TestType>(), vertical_triangle<TestType>()};
  const auto combined = crossing_triangles<TestType>();
  const auto &horizontal = inputs[0];
  const auto &vertical = inputs[1];
  const auto horizontal_faces = parity_faces(horizontal.polygons).deep_copy();
  const auto horizontal_points = parity_points(horizontal.polygons).deep_copy();
  const auto vertical_faces = parity_faces(vertical.polygons).deep_copy();
  const auto vertical_points = parity_points(vertical.polygons).deep_copy();
  const auto vertical_placement = vertical.placement.values;
  const auto combined_faces = parity_faces(combined.polygons).deep_copy();
  const auto combined_points = parity_points(combined.polygons).deep_copy();

  const auto meshes = parity_meshes_of(inputs);
  const auto mesh_arrangement = tf::cpp::mesh_arrangements_with_curves(meshes);
  const auto polygon_arrangement =
      tf::cpp::polygon_arrangements_with_curves(combined.mesh());

  REQUIRE(mesh_indices_are_valid(mesh_arrangement.mesh));
  REQUIRE(mesh_indices_are_valid(polygon_arrangement.mesh));
  REQUIRE(mesh_arrangement.mesh.size() >
          horizontal.polygons.size() + vertical.polygons.size());
  REQUIRE(polygon_arrangement.mesh.size() > combined.polygons.size());

  REQUIRE(mesh_arrangement.tag_labels.ndim() == 1);
  REQUIRE(mesh_arrangement.face_labels.ndim() == 1);
  REQUIRE(mesh_arrangement.face_labels.length() ==
          mesh_arrangement.tag_labels.length());
  REQUIRE(carrier_provenance_matches_sources(
      mesh_arrangement.mesh, mesh_arrangement.tag_labels.length(), inputs, true,
      [&](std::size_t face) { return mesh_arrangement.tag_labels[face]; },
      [&](std::size_t face) { return mesh_arrangement.face_labels[face]; }));

  std::set<std::pair<std::int32_t, std::int32_t>> source_carriers;
  for (std::size_t face = 0; face < mesh_arrangement.tag_labels.length();
       ++face)
    source_carriers.emplace(mesh_arrangement.tag_labels[face],
                            mesh_arrangement.face_labels[face]);
  CHECK(source_carriers == std::set<std::pair<std::int32_t, std::int32_t>>{
                               {0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}});

  REQUIRE(polygon_arrangement.face_labels.ndim() == 1);
  REQUIRE(carrier_provenance_matches_sources(
      polygon_arrangement.mesh, polygon_arrangement.face_labels.length(),
      std::vector<parity_operand<TestType>>{combined}, false,
      [](std::size_t) { return 0; },
      [&](std::size_t face) { return polygon_arrangement.face_labels[face]; }));

  const auto mesh_records = canonical_face_provenance(
      mesh_arrangement.mesh,
      [&](int face) {
        return mesh_arrangement.tag_labels[static_cast<std::size_t>(face)];
      },
      [&](int face) {
        return mesh_arrangement.face_labels[static_cast<std::size_t>(face)];
      });
  const auto polygon_records = canonical_face_provenance(
      polygon_arrangement.mesh,
      [&](int face) {
        const auto source_face =
            polygon_arrangement.face_labels[static_cast<std::size_t>(face)];
        return source_face < static_cast<int>(horizontal.polygons.size()) ? 0
                                                                          : 1;
      },
      [&](int face) {
        const auto source_face =
            polygon_arrangement.face_labels[static_cast<std::size_t>(face)];
        const auto source_count = static_cast<int>(horizontal.polygons.size());
        return source_face < source_count ? source_face
                                          : source_face - source_count;
      });
  CHECK(mesh_records == polygon_records);

  REQUIRE(curves_are_aligned(mesh_arrangement.curves, true));
  REQUIRE(curves_are_aligned(polygon_arrangement.curves, true));
  const auto curve_signature = tf::cpp::test::canonicalize_curves_geometry(
      mesh_arrangement.curves, tf::cpp::test::orientation_mode::ignore, true);
  CHECK(curve_signature == tf::cpp::test::canonicalize_curves_geometry(
                               polygon_arrangement.curves,
                               tf::cpp::test::orientation_mode::ignore, true));

  const auto arranged = parity_operand_of<TestType>(mesh_arrangement.mesh);
  CHECK(tf::cpp::test::within_tolerance(tf::cpp::area(arranged.mesh()),
                                        tf::cpp::area(horizontal.mesh()) +
                                            tf::cpp::area(vertical.mesh()),
                                        TestType{100}));

  // the entries read their operands and write nothing back to them
  CHECK(same_array(parity_faces(horizontal.polygons), horizontal_faces));
  CHECK(same_array(parity_points(horizontal.polygons), horizontal_points));
  CHECK(same_array(parity_faces(vertical.polygons), vertical_faces));
  CHECK(same_array(parity_points(vertical.polygons), vertical_points));
  CHECK(vertical.placement.values == vertical_placement);
  CHECK(same_array(parity_faces(combined.polygons), combined_faces));
  CHECK(same_array(parity_points(combined.polygons), combined_points));

  const std::vector<typename parity_operand<TestType>::mesh_type> singleton{
      horizontal.mesh()};
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(singleton), std::runtime_error);
}

TEMPLATE_TEST_CASE(
    "Python-parity three-input arrangements preserve N-way curve carriers",
    "[cpp][arrangement][python-parity][arrangements][three-input]", float,
    double) {
  const auto inputs = three_way_triangles<TestType>();
  const auto combined = combined_three_way_triangles<TestType>();
  const auto meshes = parity_meshes_of(inputs);
  const auto mode = tf::intersect_mode::primitives |
                    tf::intersect_mode::resolve_crossing_contours;
  const tf::intersect_config config{mode, 0.0};

  // All four Python routes have public C++ facade equivalents.
  const auto intersections = tf::cpp::intersection_curves(meshes, config);
  const auto self_intersections = tf::cpp::self_intersection_curves(
      combined.mesh(),
      {tf::intersect_mode::primitives | tf::intersect_mode::resolve_contours,
       0.0});
  const auto mesh_arrangement =
      tf::cpp::mesh_arrangements_with_curves(meshes, config);
  const auto polygon_arrangement =
      tf::cpp::polygon_arrangements_with_curves(combined.mesh(), config);

  REQUIRE(curves_are_aligned(intersections, true));
  REQUIRE(curves_are_aligned(self_intersections, true));
  REQUIRE(curves_are_aligned(mesh_arrangement.curves, true));
  REQUIRE(curves_are_aligned(polygon_arrangement.curves, true));
  REQUIRE(mesh_arrangement.mesh.size() > 3);
  REQUIRE(polygon_arrangement.mesh.size() > 3);

  REQUIRE(carrier_provenance_matches_sources(
      mesh_arrangement.mesh, mesh_arrangement.tag_labels.length(), inputs, true,
      [&](std::size_t face) { return mesh_arrangement.tag_labels[face]; },
      [&](std::size_t face) { return mesh_arrangement.face_labels[face]; }));
  REQUIRE(carrier_provenance_matches_sources(
      polygon_arrangement.mesh, polygon_arrangement.face_labels.length(),
      std::vector<parity_operand<TestType>>{combined}, false,
      [](std::size_t) { return 0; },
      [&](std::size_t face) { return polygon_arrangement.face_labels[face]; }));

  std::set<std::int32_t> tags;
  std::array<std::size_t, 3> faces_by_tag{};
  for (std::size_t face = 0; face < mesh_arrangement.tag_labels.length();
       ++face) {
    const auto tag = mesh_arrangement.tag_labels[face];
    REQUIRE(tag >= 0);
    REQUIRE(tag < 3);
    REQUIRE(mesh_arrangement.face_labels[face] == 0);
    tags.insert(tag);
    ++faces_by_tag[static_cast<std::size_t>(tag)];
  }
  CHECK(tags == std::set<std::int32_t>{0, 1, 2});
  CHECK(std::all_of(faces_by_tag.begin(), faces_by_tag.end(),
                    [](std::size_t count) { return count > 1; }));

  const auto mesh_records = canonical_face_provenance(
      mesh_arrangement.mesh,
      [&](int face) {
        return mesh_arrangement.tag_labels[static_cast<std::size_t>(face)];
      },
      [&](int face) {
        return mesh_arrangement.face_labels[static_cast<std::size_t>(face)];
      });
  const auto polygon_records = canonical_face_provenance(
      polygon_arrangement.mesh,
      [&](int face) {
        return polygon_arrangement.face_labels[static_cast<std::size_t>(face)];
      },
      [](int) { return 0; });
  CHECK(mesh_records == polygon_records);

  const auto canonical_curves = tf::cpp::test::canonicalize_curves_geometry(
      intersections, tf::cpp::test::orientation_mode::ignore, true);
  CHECK(canonical_curves ==
        tf::cpp::test::canonicalize_curves_geometry(
            self_intersections, tf::cpp::test::orientation_mode::ignore, true));
  CHECK(canonical_curves == tf::cpp::test::canonicalize_curves_geometry(
                                mesh_arrangement.curves,
                                tf::cpp::test::orientation_mode::ignore, true));
  CHECK(canonical_curves == tf::cpp::test::canonicalize_curves_geometry(
                                polygon_arrangement.curves,
                                tf::cpp::test::orientation_mode::ignore, true));

  const auto statistics = curve_statistics(intersections);
  const curve_path_statistics expected_statistics{
      6, {1, 1, 1, 1, 1, 1}, 0, 6, 7};
  CHECK(statistics == expected_statistics);
  CHECK(statistics == curve_statistics(self_intersections));
  CHECK(statistics == curve_statistics(mesh_arrangement.curves));
  CHECK(statistics == curve_statistics(polygon_arrangement.curves));

  const auto points = tf::cpp::test::arrays_of(intersections).points;
  bool contains_n_way_crossing = false;
  for (std::size_t point = 0; point < points.length() / 3; ++point) {
    const auto offset = point * 3;
    contains_n_way_crossing =
        contains_n_way_crossing ||
        (points[offset] == TestType{0} && points[offset + 1] == TestType{0} &&
         points[offset + 2] == TestType{0});
  }
  CHECK(contains_n_way_crossing);
}
