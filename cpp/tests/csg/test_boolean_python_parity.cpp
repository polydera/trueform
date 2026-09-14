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

#include "trueform/cpp/csg/make_boolean.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/topology/is_closed.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"

#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

using namespace tf::cpp::test;
TEMPLATE_TEST_CASE(
    "Python-parity booleans preserve volume topology provenance and inputs",
    "[cpp][csg][python-parity][boolean]", float, double) {
  const auto [owned_first, owned_second] = overlapping_boxes<TestType>();
  const auto first = owned_first.mesh();
  const auto second = owned_second.mesh();
  const auto first_faces = parity_faces(owned_first.polygons).deep_copy();
  const auto first_points = parity_points(owned_first.polygons).deep_copy();
  const auto second_faces = parity_faces(owned_second.polygons).deep_copy();
  const auto second_points = parity_points(owned_second.polygons).deep_copy();
  const auto second_placement = owned_second.placement.values;

  struct operation_expectation {
    tf::boolean_op operation;
    TestType volume;
  };
  const operation_expectation expectations[] = {
      {tf::boolean_op::merge, TestType{1.1875}},
      {tf::boolean_op::intersection, TestType{0.1875}},
      {tf::boolean_op::left_difference, TestType{0.8125}},
      {tf::boolean_op::right_difference, TestType{0.1875}},
  };

  for (const auto &expectation : expectations) {
    const auto result =
        tf::cpp::make_boolean(first, second, expectation.operation);
    REQUIRE(result.mesh.size() > 0);
    REQUIRE(mesh_indices_are_valid(result.mesh));
    REQUIRE(boolean_provenance_is_valid(result, owned_first, owned_second));
    const parity_operand<TestType> topology{result.mesh};
    CHECK(tf::cpp::is_closed(topology.mesh()));
    CHECK(tf::cpp::is_manifold(topology.mesh()));
    CHECK(tf::cpp::test::within_tolerance(
        std::abs(tf::cpp::signed_volume(topology.mesh())), expectation.volume,
        TestType{1000}));
  }

  const auto with_curves = tf::cpp::make_boolean_with_curves(
      first, second, tf::boolean_op::intersection);
  REQUIRE(mesh_indices_are_valid(with_curves.mesh));
  REQUIRE(boolean_provenance_is_valid(with_curves, owned_first, owned_second));
  REQUIRE(curves_are_aligned(with_curves.curves, true));
  CHECK(all_curve_paths_closed(with_curves.curves));

  const auto repeated = tf::cpp::make_boolean_with_curves(
      first, second, tf::boolean_op::intersection);
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(
            with_curves.mesh, tf::cpp::test::orientation_mode::preserve) ==
        tf::cpp::test::canonicalize_mesh_geometry(
            repeated.mesh, tf::cpp::test::orientation_mode::preserve));
  CHECK(
      tf::cpp::test::canonicalize_curves_geometry(
          with_curves.curves, tf::cpp::test::orientation_mode::ignore, true) ==
      tf::cpp::test::canonicalize_curves_geometry(
          repeated.curves, tf::cpp::test::orientation_mode::ignore, true));
  CHECK(canonical_face_provenance(
            with_curves.mesh,
            [&](int face) {
              return with_curves.labels[static_cast<std::size_t>(face)];
            },
            [&](int face) {
              return with_curves.face_labels[static_cast<std::size_t>(face)];
            }) ==
        canonical_face_provenance(
            repeated.mesh,
            [&](int face) {
              return repeated.labels[static_cast<std::size_t>(face)];
            },
            [&](int face) {
              return repeated.face_labels[static_cast<std::size_t>(face)];
            }));

  // the entries read their operands and write nothing back to them
  CHECK(same_array(parity_faces(owned_first.polygons), first_faces));
  CHECK(same_array(parity_points(owned_first.polygons), first_points));
  CHECK(same_array(parity_faces(owned_second.polygons), second_faces));
  CHECK(same_array(parity_points(owned_second.polygons), second_points));
  CHECK(owned_second.placement.values == second_placement);

  CHECK_THROWS_AS(
      tf::cpp::make_boolean(first, second, static_cast<tf::boolean_op>(99)),
      std::invalid_argument);
}
