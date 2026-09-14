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
#include "async_resolver.hpp"
#include "canonicalize.hpp"
#include "carriers.hpp"
#include "deterministic_points.hpp"
#include "fixtures.hpp"
#include "nd_array.hpp"
#include "temporary_directory.hpp"
#include "tolerances.hpp"

#include "trueform/cpp/core/async/submit.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test = tf::cpp::test;

namespace {

template <typename Real>
auto make_curves(std::vector<std::int32_t> offsets,
                 std::vector<std::int32_t> ids, std::vector<Real> points)
    -> tf::curves_buffer<tf::cpp::default_index_t, Real, 3> {
  // core's own buffer states what it was filled with, so a malformed curve is
  // stated here and refused where a signature reads it
  tf::curves_buffer<tf::cpp::default_index_t, Real, 3> out;
  test::fill_storage(out.paths_buffer().offsets_buffer(), offsets);
  test::fill_storage(out.paths_buffer().data_buffer(), ids);
  test::fill_storage(out.points_buffer().data_buffer(), points);
  return out;
}

} // namespace

TEST_CASE("common array helpers validate shape content and ownership",
          "[cpp][core][common]") {
  const auto integers = test::make_nd_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  CHECK(test::has_shape(integers, {2, 2}));
  CHECK(test::has_values(integers, {1, 2, 3, 4}));
  CHECK(test::all_finite(integers));

  const auto empty = test::make_empty_nd_array<double>({0, 3});
  CHECK(test::has_shape(empty, {0, 3}));
  CHECK(empty.empty());
  CHECK(test::all_finite(empty));

  auto source = test::make_nd_array<float>({1.0F, 2.0F}, {2});
  auto shared = source.shallow_copy();
  auto owned = source.deep_copy();
  CHECK_FALSE(test::has_independent_storage(source, shared));
  REQUIRE(test::has_independent_storage(source, owned));
  owned[0] = 9.0F;
  CHECK(source[0] == 1.0F);

  const auto non_finite = test::make_nd_array<double>(
      {0.0, std::numeric_limits<double>::infinity()}, {2});
  CHECK_FALSE(test::all_finite(non_finite));
}

TEST_CASE("common geometry fixtures provide stable public facade handles",
          "[cpp][core][common][fixtures]") {
  const auto triangle = test::triangle_mesh<float>();
  const auto patch = test::two_triangle_mesh<double>();
  const auto tetrahedron = test::tetrahedron_mesh<float>();
  const auto box = test::box_mesh<double>();
  const auto sphere = test::sphere_mesh<double>();
  const auto plane = test::plane_mesh<float>();
  const auto empty = test::empty_mesh<double>();
  const auto cloud = test::point_cloud_fixture<float>();

  CHECK(triangle.mesh().number_of_faces() == 1);
  CHECK(triangle.mesh().number_of_points() == 3);
  CHECK(patch.mesh().number_of_faces() == 2);
  CHECK(tetrahedron.mesh().number_of_faces() == 4);
  CHECK(box.mesh().number_of_faces() == 12);
  CHECK(box.mesh().number_of_points() == 8);
  CHECK(sphere.mesh().number_of_faces() == 36);
  CHECK(sphere.mesh().number_of_points() == 20);
  CHECK(plane.mesh().number_of_faces() == 2);
  CHECK(empty.mesh().number_of_faces() == 0);
  CHECK(empty.mesh().number_of_points() == 0);
  CHECK(cloud.point_cloud().number_of_points() == 5);

  const auto identity = test::identity_matrix<double>();
  const auto translation = test::translation_matrix(2.0, 3.0, 4.0);
  const auto rotate_x = test::rotation_x_matrix(0.5);
  const auto rotate_y = test::rotation_y_matrix(0.5);
  const auto rotate_z = test::rotation_z_matrix(0.5);
  const auto scale = test::scale_matrix(2.0, 3.0, 4.0);
  const auto uniform_scale = test::scale_matrix(5.0);

  CHECK(test::has_shape(identity, {4, 4}));
  CHECK(identity[0] == 1.0);
  CHECK(identity[15] == 1.0);
  CHECK(translation[3] == 2.0);
  CHECK(translation[7] == 3.0);
  CHECK(translation[11] == 4.0);
  CHECK(test::all_finite(rotate_x));
  CHECK(test::all_finite(rotate_y));
  CHECK(test::all_finite(rotate_z));
  CHECK(scale[0] == 2.0);
  CHECK(scale[5] == 3.0);
  CHECK(scale[10] == 4.0);
  CHECK(uniform_scale[10] == 5.0);
}

TEST_CASE("common edge and path canonicalizers make orientation contractual",
          "[cpp][core][common][canonicalize]") {
  const auto edges = test::make_nd_array<std::int32_t>({2, 1, 0, 3}, {2, 2});
  CHECK((test::canonicalize_oriented_edges(edges) ==
         std::vector<test::edge_signature<std::int32_t>>{{0, 3}, {2, 1}}));
  CHECK((test::canonicalize_unoriented_edges(edges) ==
         std::vector<test::edge_signature<std::int32_t>>{{0, 3}, {1, 2}}));

  const std::vector<int> face{3, 1, 2};
  const std::vector<int> rotated_face{1, 2, 3};
  const std::vector<int> reversed_face{3, 2, 1};
  CHECK(test::canonicalize_cyclic_face(face) ==
        test::canonicalize_cyclic_face(rotated_face));
  CHECK(test::canonicalize_cyclic_face(face) !=
        test::canonicalize_cyclic_face(reversed_face));
  CHECK(test::canonicalize_cyclic_face(face, test::orientation_mode::ignore) ==
        test::canonicalize_cyclic_face(reversed_face,
                                       test::orientation_mode::ignore));
  CHECK(test::canonicalize_cyclic_face(std::vector<int>{1, 2, 1}).size() == 3);

  const std::vector<int> open{1, 2, 3};
  const std::vector<int> reversed_open{3, 2, 1};
  CHECK(test::canonicalize_open_path(open) !=
        test::canonicalize_open_path(reversed_open));
  CHECK(test::canonicalize_open_path(open, test::orientation_mode::ignore) ==
        test::canonicalize_open_path(reversed_open,
                                     test::orientation_mode::ignore));

  const std::vector<int> closed{1, 2, 3, 1};
  const std::vector<int> rotated_closed{2, 3, 1, 2};
  const std::vector<int> reversed_closed{1, 3, 2, 1};
  CHECK(test::canonicalize_closed_path(closed) ==
        test::canonicalize_closed_path(rotated_closed));
  CHECK(test::canonicalize_closed_path(closed).size() == 3);
  CHECK(test::canonicalize_closed_path(closed) !=
        test::canonicalize_closed_path(reversed_closed));
  CHECK(
      test::canonicalize_closed_path(closed, test::orientation_mode::ignore) ==
      test::canonicalize_closed_path(reversed_closed,
                                     test::orientation_mode::ignore));
}

TEST_CASE("mesh signatures separate geometry from point-ID topology",
          "[cpp][core][common][canonicalize]") {
  const std::vector<float> duplicate_points{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                            1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  const auto first_duplicate =
      test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 2, 3}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0});
  const auto second_duplicate =
      test::polygons_of<tf::cpp::default_index_t, float>(
          {1, 2, 3}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK(test::canonicalize_mesh_geometry(first_duplicate) ==
        test::canonicalize_mesh_geometry(second_duplicate));
  CHECK(test::canonicalize_mesh_topology(first_duplicate) !=
        test::canonicalize_mesh_topology(second_duplicate));

  const auto joined = test::two_triangle_mesh<float>().polygons;
  const auto split = test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 1, 2, 4, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0});
  CHECK(test::canonicalize_mesh_geometry(joined) !=
        test::canonicalize_mesh_geometry(split));
  CHECK(test::canonicalize_mesh_topology(joined) !=
        test::canonicalize_mesh_topology(split));

  const auto reordered = test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 2, 3, 0, 1, 2}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
  CHECK(test::canonicalize_mesh_geometry(joined) ==
        test::canonicalize_mesh_geometry(reordered));
  CHECK(test::canonicalize_mesh_topology(joined) ==
        test::canonicalize_mesh_topology(reordered));
}

TEST_CASE("mesh signatures retain orphans and honor orientation",
          "[cpp][core][common][canonicalize]") {
  const auto oriented = test::triangle_mesh<float>().polygons;
  const auto reversed = test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 2, 1}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK(test::canonicalize_mesh_geometry(oriented) !=
        test::canonicalize_mesh_geometry(reversed));
  CHECK(test::canonicalize_mesh_topology(oriented) !=
        test::canonicalize_mesh_topology(reversed));
  CHECK(test::canonicalize_mesh_geometry(oriented,
                                         test::orientation_mode::ignore) ==
        test::canonicalize_mesh_geometry(reversed,
                                         test::orientation_mode::ignore));
  CHECK(test::canonicalize_mesh_topology(oriented,
                                         test::orientation_mode::ignore) ==
        test::canonicalize_mesh_topology(reversed,
                                         test::orientation_mode::ignore));

  const auto trailing = test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 9, 9, 9});
  CHECK(test::canonicalize_mesh_geometry(oriented) !=
        test::canonicalize_mesh_geometry(trailing));
  CHECK(test::canonicalize_mesh_topology(oriented) !=
        test::canonicalize_mesh_topology(trailing));
  CHECK_THROWS_AS(test::canonicalize_mesh_geometry(
                      trailing, test::orientation_mode::preserve, true),
                  std::invalid_argument);
  CHECK_THROWS_AS(test::canonicalize_mesh_topology(
                      trailing, test::orientation_mode::preserve, true),
                  std::invalid_argument);

  // a check reads what it is handed, so a corner the points do not reach is
  // refused by the check itself
  const auto invalid_id = test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 1, 3}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(test::canonicalize_mesh_geometry(invalid_id),
                  std::out_of_range);
  CHECK_THROWS_AS(test::canonicalize_mesh_topology(invalid_id),
                  std::out_of_range);
}

TEST_CASE("curve signatures separate geometry from point-ID topology",
          "[cpp][core][common][canonicalize]") {
  const std::vector<float> duplicate_points{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                            1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  const auto first_duplicate =
      make_curves<float>({0, 3}, {0, 2, 3}, duplicate_points);
  const auto second_duplicate =
      make_curves<float>({0, 3}, {1, 2, 3}, duplicate_points);
  CHECK(test::canonicalize_curves_geometry(first_duplicate) ==
        test::canonicalize_curves_geometry(second_duplicate));
  CHECK(test::canonicalize_curves_topology(first_duplicate) !=
        test::canonicalize_curves_topology(second_duplicate));

  const std::vector<float> points{0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                  0.0F, 1.0F, 1.0F, 0.0F};
  const auto forward = make_curves<float>({0, 3}, {0, 1, 2}, points);
  const auto backward = make_curves<float>({0, 3}, {2, 1, 0}, points);
  const auto closed = make_curves<float>({0, 4}, {0, 1, 2, 0}, points);
  CHECK(test::canonicalize_curves_geometry(forward) !=
        test::canonicalize_curves_geometry(closed));
  CHECK(test::canonicalize_curves_topology(forward) !=
        test::canonicalize_curves_topology(closed));
  CHECK(test::canonicalize_curves_geometry(forward) !=
        test::canonicalize_curves_geometry(backward));
  CHECK(test::canonicalize_curves_topology(forward) !=
        test::canonicalize_curves_topology(backward));
  CHECK(test::canonicalize_curves_geometry(forward,
                                           test::orientation_mode::ignore) ==
        test::canonicalize_curves_geometry(backward,
                                           test::orientation_mode::ignore));
  CHECK(test::canonicalize_curves_topology(forward,
                                           test::orientation_mode::ignore) ==
        test::canonicalize_curves_topology(backward,
                                           test::orientation_mode::ignore));

  const auto trailing = make_curves<float>(
      {0, 3}, {0, 1, 2},
      {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 9.0F, 9.0F, 9.0F});
  CHECK(test::canonicalize_curves_geometry(forward) !=
        test::canonicalize_curves_geometry(trailing));
  CHECK(test::canonicalize_curves_topology(forward) !=
        test::canonicalize_curves_topology(trailing));
  CHECK_THROWS_AS(test::canonicalize_curves_geometry(
                      trailing, test::orientation_mode::preserve, true),
                  std::invalid_argument);
  CHECK_THROWS_AS(test::canonicalize_curves_topology(
                      trailing, test::orientation_mode::preserve, true),
                  std::invalid_argument);
}

TEST_CASE("curve signatures reject malformed carriers before canonicalizing",
          "[cpp][core][common][canonicalize][validation]") {
  const auto reject_with_both_signatures = [](const auto &value) {
    CHECK_THROWS(test::canonicalize_curves_geometry(value));
    CHECK_THROWS(test::canonicalize_curves_topology(value));
  };

  const auto empty_offsets = make_curves<float>({}, {}, {});
  reject_with_both_signatures(empty_offsets);

  const auto two_points =
      std::vector<float>{0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  reject_with_both_signatures(make_curves<float>({1, 2}, {0, 1}, two_points));
  reject_with_both_signatures(
      make_curves<float>({0, 2, 1}, {0, 1}, two_points));
  reject_with_both_signatures(make_curves<float>({0, 1}, {0, 1}, two_points));

  const auto invalid_id = make_curves<float>({0, 2}, {2, 1}, two_points);
  CHECK_THROWS_AS(test::canonicalize_curves_geometry(invalid_id),
                  std::out_of_range);
  CHECK_THROWS_AS(test::canonicalize_curves_topology(invalid_id),
                  std::out_of_range);
}

TEST_CASE("common tolerances and point sets are scale aware and deterministic",
          "[cpp][core][common][deterministic]") {
  const auto float_tolerance = test::base_tolerance<float>();
  const auto double_tolerance = test::base_tolerance<double>();
  CHECK(float_tolerance.absolute > double_tolerance.absolute);
  CHECK(test::within_tolerance(1000.0F, 1000.005F, 1000.0F));
  CHECK_FALSE(test::within_tolerance(1.0, 1.000001));

  const auto first_grid = test::grid_points<double>(2, 2, 2, 0.5, -1.0);
  const auto second_grid = test::grid_points<double>(2, 2, 2, 0.5, -1.0);
  CHECK(test::has_shape(first_grid, {8, 3}));
  CHECK(test::has_values(
      first_grid, std::vector<double>(second_grid.begin(), second_grid.end())));
  CHECK(first_grid[0] == -1.0);
  CHECK(first_grid[3] == -0.5);

  const auto first_spiral = test::spiral_points<double>(5, 2.0, 4.0, 1.0);
  const auto second_spiral = test::spiral_points<double>(5, 2.0, 4.0, 1.0);
  CHECK(test::has_shape(first_spiral, {5, 3}));
  CHECK(
      test::has_values(first_spiral, std::vector<double>(second_spiral.begin(),
                                                         second_spiral.end())));
  CHECK(first_spiral[0] == 2.0);
  CHECK(first_spiral[1] == 0.0);
  CHECK(first_spiral[2] == 0.0);
  CHECK(first_spiral[8] == 2.0);
  CHECK(first_spiral[14] == 4.0);

  const auto first_halton = test::low_discrepancy_points<float>(8);
  const auto second_halton = test::low_discrepancy_points<float>(8);
  CHECK(test::has_shape(first_halton, {8, 3}));
  CHECK(
      test::has_values(first_halton, std::vector<float>(second_halton.begin(),
                                                        second_halton.end())));
  CHECK(first_halton[0] == 0.5F);
  CHECK(first_halton[1] == 1.0F / 3.0F);
  CHECK(first_halton[2] == 0.2F);

  constexpr auto size_limit = std::numeric_limits<std::size_t>::max();
  constexpr auto int_limit = std::numeric_limits<int>::max();
  constexpr auto has_distinct_shape_limit =
      size_limit / 3 > static_cast<std::size_t>(int_limit);
  try {
    (void)test::grid_points<double>(46341, 46341, 1);
    FAIL("grid shape overflow must reject before allocation");
  } catch (const std::overflow_error &error) {
    const auto expected = has_distinct_shape_limit
                              ? "grid point count exceeds array shape capacity"
                              : "grid coordinate count overflow";
    CHECK(std::string(error.what()) == expected);
  }

  try {
    if constexpr (has_distinct_shape_limit)
      (void)test::grid_points<double>(int_limit, int_limit, 3);
    else
      (void)test::grid_points<double>(int_limit, 2, 1);
    FAIL("grid coordinate overflow must reject before allocation");
  } catch (const std::overflow_error &error) {
    CHECK(std::string(error.what()) == "grid coordinate count overflow");
  }
}

TEST_CASE("common async and temporary-directory helpers own shared state",
          "[cpp][core][common][async][filesystem]") {
  test::counting_resolver resolver;
  CHECK(test::resolve_once(resolver, [](auto counting) {
          return tf::cpp::async::submit(counting, [] { return 42; });
        }) == 42);
  CHECK(resolver.submissions() == 1);

  CHECK(tf::cpp::async::submit<int>(tf::cpp::async::future_resolver{}, [] {
          return 17;
        }).get() == 17);

  std::filesystem::path removed_path;
  {
    test::temporary_directory directory("trueform common helpers");
    removed_path = directory.path();
    REQUIRE(std::filesystem::is_directory(removed_path));
    std::ofstream(removed_path / "owned.txt") << "owned";
    CHECK(std::filesystem::is_regular_file(removed_path / "owned.txt"));
  }
  CHECK_FALSE(std::filesystem::exists(removed_path));
}
