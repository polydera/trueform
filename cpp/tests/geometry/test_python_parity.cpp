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
#include "carriers.hpp"
#include "nd_array.hpp"

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/chamfer_error.hpp"
#include "trueform/cpp/geometry/fit_rigid.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_cylinder_mesh.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/max_edge_length.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/min_edge_length.hpp"
#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/geometry/point_normals.hpp"
#include "trueform/cpp/geometry/positively_oriented.hpp"
#include "trueform/cpp/geometry/reverse_winding.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/geometry/triangulate.hpp"
#include "trueform/cpp/topology.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr double parity_pi = 3.141592653589793238462643383279502884;

using parity_index = tf::cpp::default_index_t;

template <typename Real>
using parity_owned = tf::cpp::test::owned_mesh<parity_index, Real>;

template <typename Real>
using parity_result = tf::polygons_buffer<parity_index, Real, 3, 3>;

template <typename Real> auto tolerance() -> double {
  return std::is_same_v<Real, float> ? 2e-5 : 1e-11;
}

/// A result is core's storage; a caller that wants to ask it anything holds it
/// beside a cache of its own, which is the assembly.
template <typename Real>
auto held(parity_result<Real> value) -> parity_owned<Real> {
  parity_owned<Real> owned;
  owned.polygons = std::move(value);
  return owned;
}

template <typename Real>
auto coordinates_of(const parity_owned<Real> &owned)
    -> const tf::buffer<Real> & {
  return owned.polygons.points_buffer().data_buffer();
}

template <typename Real>
auto corners_of(const parity_owned<Real> &owned)
    -> const tf::buffer<parity_index> & {
  return owned.polygons.faces_buffer().data_buffer();
}

template <typename Real>
auto twice_triangle_area(const parity_owned<Real> &owned, std::size_t face)
    -> double {
  const auto &faces = corners_of(owned);
  const auto &points = coordinates_of(owned);
  const auto face_offset = face * 3;
  const auto a = static_cast<std::size_t>(3 * faces[face_offset]);
  const auto b = static_cast<std::size_t>(3 * faces[face_offset + 1]);
  const auto c = static_cast<std::size_t>(3 * faces[face_offset + 2]);
  const auto abx = static_cast<double>(points[b] - points[a]);
  const auto aby = static_cast<double>(points[b + 1] - points[a + 1]);
  const auto abz = static_cast<double>(points[b + 2] - points[a + 2]);
  const auto acx = static_cast<double>(points[c] - points[a]);
  const auto acy = static_cast<double>(points[c + 1] - points[a + 1]);
  const auto acz = static_cast<double>(points[c + 2] - points[a + 2]);
  const auto cross_x = aby * acz - abz * acy;
  const auto cross_y = abz * acx - abx * acz;
  const auto cross_z = abx * acy - aby * acx;
  return std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
}

template <typename Real>
auto check_triangle_geometry(const parity_owned<Real> &owned, bool closed)
    -> void {
  const auto face_count =
      static_cast<std::size_t>(owned.polygons.faces_buffer().size());
  const auto point_count =
      static_cast<std::size_t>(owned.polygons.points_buffer().size());
  REQUIRE(face_count > 0);
  REQUIRE(point_count > 0);
  const auto &faces = corners_of(owned);
  for (std::size_t face = 0; face != face_count; ++face) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const auto index = faces[face * 3 + corner];
      CHECK(index >= 0);
      CHECK(static_cast<std::size_t>(index) < point_count);
    }
    CHECK(twice_triangle_area(owned, face) > 1e-10);
  }
  CHECK(tf::cpp::is_manifold(owned.mesh()));
  CHECK(tf::cpp::is_closed(owned.mesh()) == closed);
}

template <typename Real> auto planar_square() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<parity_index, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Real> auto negative_tetrahedron() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<parity_index, Real>(
      {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{0}, Real{1}})};
}

template <typename Real> auto similarity_transform() -> std::array<Real, 16> {
  return {Real{0}, Real{-2},  Real{0}, Real{100}, Real{2}, Real{0},
          Real{0}, Real{-50}, Real{0}, Real{0},   Real{2}, Real{25},
          Real{0}, Real{0},   Real{0}, Real{1}};
}

template <typename Real>
auto cloud(std::initializer_list<Real> values)
    -> tf::cpp::test::owned_point_cloud<Real> {
  tf::cpp::test::owned_point_cloud<Real> owned;
  owned.points = tf::cpp::test::points_of<Real>(values);
  return owned;
}

template <typename Real>
auto apply_matrix(const tf::cpp::nd_array<Real> &matrix,
                  const tf::buffer<Real> &points) -> std::vector<Real> {
  std::vector<Real> output(points.size());
  for (std::size_t point = 0; point * 3 < points.size(); ++point) {
    const auto offset = point * 3;
    for (std::size_t row = 0; row < 3; ++row) {
      const auto matrix_offset = 4 * row;
      output[offset + row] = matrix[matrix_offset] * points[offset] +
                             matrix[matrix_offset + 1] * points[offset + 1] +
                             matrix[matrix_offset + 2] * points[offset + 2] +
                             matrix[matrix_offset + 3];
    }
  }
  return output;
}

template <typename Real>
auto check_rigid_matrix(const tf::cpp::nd_array<Real> &matrix) -> void {
  REQUIRE((matrix.raw_shape() == tf::small_vector<int, 3>{4, 4}));
  const auto margin = tolerance<Real>();
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      auto dot = 0.0;
      for (int coordinate = 0; coordinate < 3; ++coordinate)
        dot += static_cast<double>(
                   matrix[static_cast<std::size_t>(4 * first + coordinate)]) *
               static_cast<double>(
                   matrix[static_cast<std::size_t>(4 * second + coordinate)]);
      CHECK(dot == Catch::Approx(first == second ? 1.0 : 0.0).margin(margin));
    }
  }
  const auto determinant = static_cast<double>(matrix[0]) *
                               (static_cast<double>(matrix[5]) * matrix[10] -
                                static_cast<double>(matrix[6]) * matrix[9]) -
                           static_cast<double>(matrix[1]) *
                               (static_cast<double>(matrix[4]) * matrix[10] -
                                static_cast<double>(matrix[6]) * matrix[8]) +
                           static_cast<double>(matrix[2]) *
                               (static_cast<double>(matrix[4]) * matrix[9] -
                                static_cast<double>(matrix[5]) * matrix[8]);
  CHECK(determinant == Catch::Approx(1.0).margin(margin));
  CHECK(matrix[12] == Real{0});
  CHECK(matrix[13] == Real{0});
  CHECK(matrix[14] == Real{0});
  CHECK(matrix[15] == Real{1});
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python parity mesh primitives are nondegenerate with analytic measures",
    "[cpp][geometry][python-parity][mesh-primitives]", float, double) {
  const auto sphere =
      held(tf::cpp::make_sphere_mesh<parity_index>(TestType{2}, 32, 48));
  const auto cylinder = held(tf::cpp::make_cylinder_mesh<parity_index>(
      TestType{1.5}, TestType{3}, 64));
  const auto box = held(tf::cpp::make_box_mesh<parity_index>(
      TestType{2}, TestType{3}, TestType{4}, 2, 3, 4));
  const auto plane = held(
      tf::cpp::make_plane_mesh<parity_index>(TestType{6}, TestType{4}, 5, 3));

  check_triangle_geometry(sphere, true);
  check_triangle_geometry(cylinder, true);
  check_triangle_geometry(box, true);
  check_triangle_geometry(plane, false);

  CHECK(tf::cpp::area(sphere.mesh()) ==
        Catch::Approx(16.0 * parity_pi).epsilon(0.01));
  CHECK(tf::cpp::signed_volume(sphere.mesh()) ==
        Catch::Approx((32.0 / 3.0) * parity_pi).epsilon(0.01));
  CHECK(tf::cpp::area(cylinder.mesh()) ==
        Catch::Approx(13.5 * parity_pi).epsilon(0.01));
  CHECK(tf::cpp::signed_volume(cylinder.mesh()) ==
        Catch::Approx(6.75 * parity_pi).epsilon(0.01));
  CHECK(tf::cpp::area(box.mesh()) ==
        Catch::Approx(52.0).margin(tolerance<TestType>()));
  CHECK(tf::cpp::signed_volume(box.mesh()) ==
        Catch::Approx(24.0).margin(tolerance<TestType>()));
  CHECK(tf::cpp::area(plane.mesh()) ==
        Catch::Approx(24.0).margin(tolerance<TestType>()));
  CHECK(tf::cpp::signed_volume(plane.mesh()) ==
        Catch::Approx(0.0).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "Python parity measurements preserve winding and similarity invariants",
    "[cpp][geometry][python-parity][measurements][transformation]", float,
    double) {
  auto owned = held(tf::cpp::make_box_mesh<parity_index>(
      TestType{2}, TestType{3}, TestType{4}));
  const auto base_area = tf::cpp::area(owned.mesh());
  const auto base_volume = tf::cpp::signed_volume(owned.mesh());
  const auto base_mean = tf::cpp::mean_edge_length(owned.mesh());
  const auto base_min = tf::cpp::min_edge_length(owned.mesh());
  const auto base_max = tf::cpp::max_edge_length(owned.mesh());
  REQUIRE(base_volume > TestType{0});

  const auto reversed = held(tf::cpp::reverse_winding(owned.mesh()));
  CHECK(tf::cpp::area(reversed.mesh()) ==
        Catch::Approx(base_area).margin(tolerance<TestType>()));
  CHECK(tf::cpp::signed_volume(reversed.mesh()) ==
        Catch::Approx(-base_volume).margin(tolerance<TestType>()));
  CHECK(tf::cpp::mean_edge_length(reversed.mesh()) ==
        Catch::Approx(base_mean).margin(tolerance<TestType>()));
  CHECK(tf::cpp::min_edge_length(reversed.mesh()) ==
        Catch::Approx(base_min).margin(tolerance<TestType>()));
  CHECK(tf::cpp::max_edge_length(reversed.mesh()) ==
        Catch::Approx(base_max).margin(tolerance<TestType>()));

  owned.place(similarity_transform<TestType>());
  CHECK(tf::cpp::area(owned.mesh()) ==
        Catch::Approx(4 * base_area).margin(10 * tolerance<TestType>()));
  CHECK(tf::cpp::signed_volume(owned.mesh()) ==
        Catch::Approx(8 * base_volume).margin(10 * tolerance<TestType>()));
  CHECK(tf::cpp::mean_edge_length(owned.mesh()) ==
        Catch::Approx(2 * base_mean).margin(10 * tolerance<TestType>()));
  CHECK(tf::cpp::min_edge_length(owned.mesh()) ==
        Catch::Approx(2 * base_min).margin(10 * tolerance<TestType>()));
  CHECK(tf::cpp::max_edge_length(owned.mesh()) ==
        Catch::Approx(2 * base_max).margin(10 * tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("Python parity normals have canonical values and shapes",
                   "[cpp][geometry][python-parity][normals]", float, double) {
  const auto owned = planar_square<TestType>();

  const auto face_normals = tf::cpp::normals(owned.mesh());
  const auto point_normals = tf::cpp::point_normals(owned.mesh());
  REQUIRE((face_normals.raw_shape() == tf::small_vector<int, 3>{2, 3}));
  REQUIRE((point_normals.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  for (int normal = 0; normal < face_normals.shape_at(0); ++normal) {
    const auto offset = static_cast<std::size_t>(3 * normal);
    CHECK(face_normals[offset] ==
          Catch::Approx(0.0).margin(tolerance<TestType>()));
    CHECK(face_normals[offset + 1] ==
          Catch::Approx(0.0).margin(tolerance<TestType>()));
    CHECK(face_normals[offset + 2] ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
  }
  for (int normal = 0; normal < point_normals.shape_at(0); ++normal) {
    const auto offset = static_cast<std::size_t>(3 * normal);
    CHECK(point_normals[offset] ==
          Catch::Approx(0.0).margin(tolerance<TestType>()));
    CHECK(point_normals[offset + 1] ==
          Catch::Approx(0.0).margin(tolerance<TestType>()));
    CHECK(point_normals[offset + 2] ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
  }

  // a normal is the caller's value, so a second ask states a second array
  CHECK(tf::cpp::normals(owned.mesh()).raw_owner() != face_normals.raw_owner());
}

TEMPLATE_TEST_CASE(
    "Python parity positive orientation flips a consistent negative copy",
    "[cpp][geometry][python-parity][orientation][ownership]", float, double) {
  const auto input = negative_tetrahedron<TestType>();
  const auto &source_corners = corners_of(input);
  const std::vector<parity_index> original_faces(source_corners.begin(),
                                                 source_corners.end());
  const auto *point_data = coordinates_of(input).data();
  const auto before = tf::cpp::signed_volume(input.mesh());
  REQUIRE(before < TestType{0});

  const auto output = held(tf::cpp::positively_oriented(input.mesh(), true));
  CHECK(tf::cpp::signed_volume(output.mesh()) ==
        Catch::Approx(-before).margin(tolerance<TestType>()));
  CHECK(corners_of(output).data() != source_corners.data());
  CHECK(coordinates_of(output).data() != point_data);
  for (std::size_t index = 0; index != original_faces.size(); ++index)
    CHECK(source_corners[index] == original_faces[index]);
  CHECK(tf::cpp::signed_volume(input.mesh()) ==
        Catch::Approx(before).margin(tolerance<TestType>()));
  CHECK(coordinates_of(output)[3] == TestType{1});
  CHECK(tf::cpp::signed_volume(output.mesh()) > TestType{0});
}

TEMPLATE_TEST_CASE(
    "Python parity triangulation covers a concave polygon without degeneracy",
    "[cpp][geometry][python-parity][triangulate]", float, double) {
  auto polygon = tf::cpp::test::make_nd_array<TestType>(
      {TestType{0}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}},
      {6, 3});
  const auto output =
      held(tf::cpp::triangulate<parity_index, TestType>(polygon));

  REQUIRE(output.polygons.faces_buffer().size() == 4);
  REQUIRE(output.polygons.points_buffer().size() == 6);
  CHECK(static_cast<const void *>(coordinates_of(output).data()) !=
        static_cast<const void *>(polygon.raw_data()));
  CHECK(tf::cpp::area(output.mesh()) ==
        Catch::Approx(3.0).margin(tolerance<TestType>()));
  check_triangle_geometry(output, false);

  const auto first_coordinate = coordinates_of(output)[0];
  polygon[0] = TestType{100};
  polygon.destroy();
  CHECK(coordinates_of(output)[0] == first_coordinate);
}

TEMPLATE_TEST_CASE(
    "Python parity rigid registration recovers rotation and translation",
    "[cpp][geometry][python-parity][registration][rigid]", float, double) {
  const auto source = cloud<TestType>(
      {TestType{0}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{3}, TestType{1}, TestType{2}, TestType{1}});
  const auto target = cloud<TestType>(
      {TestType{2}, TestType{-3}, TestType{1}, TestType{2}, TestType{-1},
       TestType{1}, TestType{1}, TestType{-3}, TestType{1}, TestType{2},
       TestType{-3}, TestType{4}, TestType{0}, TestType{-2}, TestType{2}});

  const auto result =
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud());
  check_rigid_matrix(result);
  const auto transformed = apply_matrix(result, source.points.data_buffer());
  const auto &target_points = target.points.data_buffer();
  REQUIRE(transformed.size() == target_points.size());
  for (std::size_t index = 0; index != transformed.size(); ++index)
    CHECK(
        transformed[index] ==
        Catch::Approx(target_points[index]).margin(10 * tolerance<TestType>()));
  CHECK(result[3] == Catch::Approx(2.0).margin(10 * tolerance<TestType>()));
  CHECK(result[7] == Catch::Approx(-3.0).margin(10 * tolerance<TestType>()));
  CHECK(result[11] == Catch::Approx(1.0).margin(10 * tolerance<TestType>()));
  CHECK(result.is_valid());
}

TEMPLATE_TEST_CASE(
    "Python parity Chamfer error is directional for unequal point sets",
    "[cpp][geometry][python-parity][registration][chamfer]", float, double) {
  const auto singleton =
      cloud<TestType>({TestType{0}, TestType{0}, TestType{0}});
  const auto pair = cloud<TestType>({TestType{1}, TestType{0}, TestType{0},
                                     TestType{2}, TestType{0}, TestType{0}});

  const auto singleton_to_pair =
      tf::cpp::chamfer_error(singleton.point_cloud(), pair.point_cloud());
  const auto pair_to_singleton =
      tf::cpp::chamfer_error(pair.point_cloud(), singleton.point_cloud());
  CHECK(singleton_to_pair == Catch::Approx(1.0).margin(tolerance<TestType>()));
  CHECK(pair_to_singleton == Catch::Approx(1.5).margin(tolerance<TestType>()));
  CHECK(singleton_to_pair != pair_to_singleton);
}
