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

#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/remesh/async/decimated.hpp"
#include "trueform/cpp/remesh/async/isotropic_remeshed.hpp"
#include "trueform/cpp/remesh/async/simplified.hpp"
#include "trueform/cpp/remesh/decimated.hpp"
#include "trueform/cpp/remesh/isotropic_remeshed.hpp"
#include "trueform/cpp/remesh/simplified.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

template <typename Index, typename Real>
using parity_owned = tf::cpp::test::owned_mesh<Index, Real>;

template <typename Index, typename Real>
using parity_result = tf::polygons_buffer<Index, Real, 3, 3>;

template <typename Index, typename Real>
auto parity_sphere(int stacks) -> parity_owned<Index, Real> {
  parity_owned<Index, Real> owned;
  owned.polygons =
      tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, stacks, stacks);
  return owned;
}

template <typename Index, typename Real>
auto parity_box(Real width, Real height, Real depth)
    -> parity_owned<Index, Real> {
  parity_owned<Index, Real> owned;
  owned.polygons = tf::cpp::make_box_mesh<Index, Real>(width, height, depth);
  return owned;
}

template <typename Index, typename Real>
auto parity_subdivided_box(Real width, Real height, Real depth,
                           std::int32_t ticks) -> parity_owned<Index, Real> {
  parity_owned<Index, Real> owned;
  owned.polygons = tf::cpp::make_box_mesh<Index, Real>(width, height, depth,
                                                       ticks, ticks, ticks);
  return owned;
}

template <typename Index, typename Real>
auto require_valid_triangle_mesh(const parity_result<Index, Real> &mesh)
    -> void {
  REQUIRE(mesh.size() > 0);
  REQUIRE(mesh.points_buffer().size() > 0);
  REQUIRE(mesh.faces_buffer().data_buffer().size() == mesh.size() * 3);
  for (const auto point_id : mesh.faces_buffer().data_buffer()) {
    CHECK(point_id >= 0);
    CHECK(static_cast<std::size_t>(point_id) < mesh.points_buffer().size());
  }
}

template <typename Index, typename Real>
auto labels_by_face_centroid_x(const parity_result<Index, Real> &mesh)
    -> tf::cpp::nd_array<std::int32_t> {
  auto labels = tf::cpp::test::make_nd_array<std::int32_t>(
      std::vector<std::int32_t>(mesh.size(), 0),
      {static_cast<int>(mesh.size())});

  const auto &corners = mesh.faces_buffer().data_buffer();
  const auto &coordinates = mesh.points_buffer().data_buffer();
  for (std::size_t face = 0; face < mesh.size(); ++face) {
    Real centroid_x{};
    for (std::size_t corner = 0; corner < 3; ++corner)
      centroid_x +=
          coordinates[static_cast<std::size_t>(corners[face * 3 + corner]) * 3];
    labels[face] = centroid_x > Real{0} ? 1 : 0;
  }
  return labels;
}

template <typename Index, typename Real>
auto check_region_alignment(const tf::cpp::remesh_result<Index, Real> &result)
    -> void {
  require_valid_triangle_mesh<Index, Real>(result.mesh);
  REQUIRE(result.regions.ndim() == 1);
  REQUIRE(result.regions.length() == result.mesh.size());

  std::set<std::int32_t> present;
  std::array<Real, 2> centroid_sums{};
  std::array<int, 2> label_counts{};
  const auto &corners = result.mesh.faces_buffer().data_buffer();
  const auto &coordinates = result.mesh.points_buffer().data_buffer();
  for (std::size_t face = 0; face < result.mesh.size(); ++face) {
    const auto label = result.regions[face];
    present.insert(label);
    REQUIRE((label == 0 || label == 1));

    Real centroid_x{};
    for (std::size_t corner = 0; corner < 3; ++corner)
      centroid_x +=
          coordinates[static_cast<std::size_t>(corners[face * 3 + corner]) * 3];
    centroid_sums[static_cast<std::size_t>(label)] += centroid_x / Real{3};
    ++label_counts[static_cast<std::size_t>(label)];
    const auto boundary_tolerance = Real{1e-4};
    if (label == 0)
      CHECK(centroid_x <= boundary_tolerance);
    else
      CHECK(centroid_x >= -boundary_tolerance);
  }
  CHECK(present == std::set<std::int32_t>{0, 1});
  REQUIRE(label_counts[0] > 0);
  REQUIRE(label_counts[1] > 0);
  CHECK(centroid_sums[0] / static_cast<Real>(label_counts[0]) < Real{0});
  CHECK(centroid_sums[1] / static_cast<Real>(label_counts[1]) > Real{0});
}

template <typename Index, typename Real>
auto count_box_corners(const parity_result<Index, Real> &mesh) -> int {
  const auto &coordinates = mesh.points_buffer().data_buffer();
  const auto tolerance =
      std::is_same_v<Real, float> ? Real{1e-5F} : Real{1e-10};
  int count = 0;
  for (const auto sx : {Real{-1}, Real{1}})
    for (const auto sy : {Real{-1}, Real{1}})
      for (const auto sz : {Real{-1}, Real{1}}) {
        bool found = false;
        for (std::size_t point = 0; point < mesh.points_buffer().size();
             ++point) {
          const auto offset = point * 3;
          found =
              found || (std::abs(coordinates[offset] - sx) <= tolerance &&
                        std::abs(coordinates[offset + 1] - sy) <= tolerance &&
                        std::abs(coordinates[offset + 2] - sz) <= tolerance);
        }
        count += found ? 1 : 0;
      }
  return count;
}

template <typename Index, typename Real> struct remesh_case {
  using real_type = Real;
  using index_type = Index;
};

using remesh_matrix = std::tuple<
    remesh_case<std::int32_t, float>, remesh_case<std::int64_t, float>,
    remesh_case<std::int32_t, double>, remesh_case<std::int64_t, double>>;

template <typename Mesh, typename = void>
struct has_simplified : std::false_type {};

template <typename Mesh>
struct has_simplified<
    Mesh, std::void_t<decltype(tf::cpp::simplified(std::declval<Mesh>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_simplified : std::false_type {};

template <typename Mesh>
struct has_async_simplified<
    Mesh,
    std::void_t<decltype(tf::cpp::async::simplified(std::declval<Mesh>()))>>
    : std::true_type {};

static_assert(
    has_simplified<const tf::cpp::mesh<std::int32_t, float, 3> &>::value);
static_assert(
    has_simplified<const tf::cpp::mesh<std::int64_t, double, 3> &>::value);
static_assert(
    !has_simplified<const tf::cpp::mesh<std::int32_t, float, 2> &>::value);
static_assert(!has_async_simplified<
              const tf::cpp::mesh<std::int64_t, double, 2> &>::value);

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "Python parity simplify reduces geometry within its error budget",
    "[cpp][remesh][python-parity][simplify]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto sphere = parity_sphere<Index, Real>(20);
  const std::vector<Index> source_faces(
      sphere.polygons.faces_buffer().data_buffer().begin(),
      sphere.polygons.faces_buffer().data_buffer().end());
  const std::vector<Real> source_points(
      sphere.polygons.points_buffer().data_buffer().begin(),
      sphere.polygons.points_buffer().data_buffer().end());
  const auto original_faces = sphere.polygons.size();
  const auto original_volume = tf::cpp::signed_volume(sphere.mesh());

  tf::simplify_config<Real> config(Real{0.01});

  const auto result = tf::cpp::simplified(sphere.mesh(), config);
  static_assert(
      std::is_same_v<decltype(result.mesh), parity_result<Index, Real>>);
  require_valid_triangle_mesh<Index, Real>(result.mesh);
  CHECK(result.mesh.size() < original_faces);
  CHECK(result.regions.is_valid());
  CHECK(result.regions.empty());

  parity_owned<Index, Real> simplified{result.mesh};
  const auto volume_ratio =
      tf::cpp::signed_volume(simplified.mesh()) / original_volume;
  CHECK(volume_ratio > Real{0.5});
  CHECK(volume_ratio < Real{1.5});
  CHECK(std::equal(sphere.polygons.faces_buffer().data_buffer().begin(),
                   sphere.polygons.faces_buffer().data_buffer().end(),
                   source_faces.begin(), source_faces.end()));
  CHECK(std::equal(sphere.polygons.points_buffer().data_buffer().begin(),
                   sphere.polygons.points_buffer().data_buffer().end(),
                   source_points.begin(), source_points.end()));

  auto box = parity_subdivided_box<Index, Real>(Real{2}, Real{3}, Real{4}, 4);
  const auto box_faces = box.polygons.size();
  const auto flat_result = tf::cpp::simplified(box.mesh());
  require_valid_triangle_mesh<Index, Real>(flat_result.mesh);
  CHECK(flat_result.mesh.size() <= box_faces);
}

TEMPLATE_LIST_TEST_CASE(
    "Python parity simplify accepts every numeric and feature option",
    "[cpp][remesh][python-parity][simplify][options]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto source = parity_sphere<Index, Real>(15);

  tf::simplify_config<Real> config;
  config.error_rel = Real{0.005};
  config.optimize_iterations = 2;
  config.min_quality = Real{0.2};
  config.preserve_boundary = false;
  config.stabilizer = 1e-3;
  config.parallel = true;
  config.feature_angle =
      decltype(config.feature_angle)(static_cast<Real>(30.0 * pi / 180.0));
  config.feature_weight = Real{100};

  const auto result = tf::cpp::simplified(source.mesh(), config);
  require_valid_triangle_mesh<Index, Real>(result.mesh);
  CHECK(result.mesh.size() <= source.polygons.size());
}

TEMPLATE_LIST_TEST_CASE(
    "Python parity remesh regions survive and remain face aligned",
    "[cpp][remesh][python-parity][regions]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto source =
      parity_subdivided_box<Index, Real>(Real{2}, Real{3}, Real{4}, 6);
  auto regions = labels_by_face_centroid_x<Index, Real>(source.polygons);
  const std::vector<Index> source_faces(
      source.polygons.faces_buffer().data_buffer().begin(),
      source.polygons.faces_buffer().data_buffer().end());
  const std::vector<Real> source_points(
      source.polygons.points_buffer().data_buffer().begin(),
      source.polygons.points_buffer().data_buffer().end());
  std::vector<std::int32_t> original_regions(regions.begin(), regions.end());

  tf::simplify_config<Real> simplify;
  tf::decimate_config<Real> decimate;
  tf::isotropic_remesh_config<Real> isotropic(
      Real{2} * tf::cpp::mean_edge_length(source.mesh()));

  auto simplified = tf::cpp::simplified(source.mesh(), simplify, regions);
  auto decimated =
      tf::cpp::decimated(source.mesh(), Real{0.5}, decimate, regions);
  auto remeshed =
      tf::cpp::isotropic_remeshed(source.mesh(), isotropic, regions);

  check_region_alignment<Index, Real>(simplified);
  check_region_alignment<Index, Real>(decimated);
  check_region_alignment<Index, Real>(remeshed);
  CHECK(std::equal(source.polygons.faces_buffer().data_buffer().begin(),
                   source.polygons.faces_buffer().data_buffer().end(),
                   source_faces.begin(), source_faces.end()));
  CHECK(std::equal(source.polygons.points_buffer().data_buffer().begin(),
                   source.polygons.points_buffer().data_buffer().end(),
                   source_points.begin(), source_points.end()));
  CHECK(std::equal(regions.begin(), regions.end(), original_regions.begin(),
                   original_regions.end()));

  REQUIRE_FALSE(simplified.regions.empty());
  const auto source_first_point =
      source.polygons.points_buffer().data_buffer()[0];
  simplified.mesh.points_buffer().data_buffer()[0] += Real{17};
  simplified.regions[0] = 99;
  CHECK(source.polygons.points_buffer().data_buffer()[0] == source_first_point);
  CHECK(regions[0] == original_regions[0]);
}

TEMPLATE_LIST_TEST_CASE(
    "Python parity feature angle locks sharp corners without regions",
    "[cpp][remesh][python-parity][feature-angle]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto source =
      parity_subdivided_box<Index, Real>(Real{2}, Real{2}, Real{2}, 16);
  const std::vector<Index> source_faces(
      source.polygons.faces_buffer().data_buffer().begin(),
      source.polygons.faces_buffer().data_buffer().end());
  const std::vector<Real> source_points(
      source.polygons.points_buffer().data_buffer().begin(),
      source.polygons.points_buffer().data_buffer().end());
  const auto feature_angle =
      tf::rad<Real>(static_cast<Real>(30.0 * pi / 180.0));

  tf::simplify_config<Real> plain_simplify(Real{0.05});
  plain_simplify.parallel = false;
  auto feature_simplify = plain_simplify;
  feature_simplify.feature_angle = feature_angle;

  tf::decimate_config<Real> feature_decimate;
  feature_decimate.parallel = false;
  feature_decimate.feature_angle = feature_angle;

  tf::isotropic_remesh_config<Real> feature_isotropic(
      Real{2} * tf::cpp::mean_edge_length(source.mesh()));
  feature_isotropic.parallel = false;
  feature_isotropic.feature_angle = feature_angle;

  const auto plain_result = tf::cpp::simplified(source.mesh(), plain_simplify);
  const auto simplified = tf::cpp::simplified(source.mesh(), feature_simplify);
  const auto decimated =
      tf::cpp::decimated(source.mesh(), Real{0.05}, feature_decimate);
  const auto remeshed =
      tf::cpp::isotropic_remeshed(source.mesh(), feature_isotropic);

  require_valid_triangle_mesh<Index, Real>(plain_result.mesh);
  CHECK(count_box_corners<Index, Real>(plain_result.mesh) < 8);
  for (const auto *result : {&simplified, &decimated, &remeshed}) {
    require_valid_triangle_mesh<Index, Real>(result->mesh);
    CHECK(result->regions.is_valid());
    CHECK(result->regions.empty());
    CHECK(count_box_corners<Index, Real>(result->mesh) == 8);
  }
  CHECK(plain_result.regions.is_valid());
  CHECK(plain_result.regions.empty());
  CHECK(std::equal(source.polygons.faces_buffer().data_buffer().begin(),
                   source.polygons.faces_buffer().data_buffer().end(),
                   source_faces.begin(), source_faces.end()));
  CHECK(std::equal(source.polygons.points_buffer().data_buffer().begin(),
                   source.polygons.points_buffer().data_buffer().end(),
                   source_points.begin(), source_points.end()));
}

TEMPLATE_LIST_TEST_CASE(
    "typed remesh accepts the native supplied-empty label convenience",
    "[cpp][remesh][python-parity][regions][empty-native]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto source = parity_box<Index, Real>(Real{2}, Real{3}, Real{4});
  auto empty = tf::cpp::test::make_nd_array<std::int32_t>({}, {0});

  tf::isotropic_remesh_config<Real> isotropic(Real{1}, 0);
  tf::simplify_config<Real> simplify;
  simplify.iterations = 0;
  const auto decimated = tf::cpp::decimated(source.mesh(), Real{1},
                                            tf::decimate_config<Real>{}, empty);
  const auto remeshed =
      tf::cpp::isotropic_remeshed(source.mesh(), isotropic, empty);
  const auto simplified = tf::cpp::simplified(source.mesh(), simplify, empty);

  for (const auto *result : {&decimated, &remeshed, &simplified}) {
    CHECK(result->regions.is_valid());
    CHECK(result->regions.empty());
  }
}

TEMPLATE_LIST_TEST_CASE("typed remesh futures retain the regions they carry",
                        "[cpp][remesh][python-parity][async]", remesh_matrix) {
  // THE ASYNC ARRAY LAW: a job carries the HANDLE, so the caller may release
  // its own after dispatch and the storage stays; what the caller must not do
  // before the future completes is write through it.
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using result_type = tf::cpp::remesh_result<Index, Real>;
  auto source = parity_box<Index, Real>(Real{2}, Real{3}, Real{4});
  auto regions = tf::cpp::test::make_nd_array<std::int32_t>(
      std::vector<std::int32_t>(source.polygons.size(), 23),
      {static_cast<int>(source.polygons.size())});

  tf::decimate_config<Real> decimate;
  decimate.parallel = false;
  tf::isotropic_remesh_config<Real> isotropic(Real{1}, 0);
  isotropic.parallel = false;
  tf::simplify_config<Real> simplify;
  simplify.iterations = 0;
  simplify.parallel = false;

  auto decimated =
      tf::cpp::async::decimated(source.mesh(), Real{1}, decimate, regions);
  auto remeshed =
      tf::cpp::async::isotropic_remeshed(source.mesh(), isotropic, regions);
  auto simplified =
      tf::cpp::async::simplified(source.mesh(), simplify, regions);
  static_assert(std::is_same_v<decltype(decimated), std::future<result_type>>);
  static_assert(std::is_same_v<decltype(remeshed), std::future<result_type>>);
  static_assert(std::is_same_v<decltype(simplified), std::future<result_type>>);

  regions.destroy();

  for (auto *pending : {&decimated, &remeshed, &simplified}) {
    const auto result = pending->get();
    REQUIRE_FALSE(result.regions.empty());
    CHECK(result.regions[0] == 23);
  }
}

TEMPLATE_LIST_TEST_CASE(
    "typed remesh validation is independent of index storage",
    "[cpp][remesh][python-parity][validation]", remesh_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto source = parity_box<Index, Real>(Real{2}, Real{3}, Real{4});
  auto wrong_regions = tf::cpp::test::make_nd_array<std::int32_t>(
      std::vector<std::int32_t>(source.polygons.size() - 1, 0),
      {static_cast<int>(source.polygons.size() - 1)});

  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), {}, wrong_regions),
                  std::invalid_argument);
  auto malformed_empty = tf::cpp::test::make_nd_array<std::int32_t>({}, {0, 2});
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), {}, malformed_empty),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::decimated(source.mesh(), Real{0.5}, {}, malformed_empty),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isotropic_remeshed(source.mesh(),
                                  tf::isotropic_remesh_config<Real>{Real{1}, 0},
                                  malformed_empty),
      std::invalid_argument);
  tf::simplify_config<Real> invalid_config;
  invalid_config.error_rel = Real{-1};
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), invalid_config),
                  std::invalid_argument);

  // the cache owns the indices for the reading it answers, so a read refuses
  // what a restatement of the points put out of reach
  auto &coordinates = source.polygons.points_buffer().data_buffer();
  coordinates.allocate(6);
  std::fill(coordinates.begin(), coordinates.end(), Real{0});
  source.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::decimated(source.mesh(), Real{0.5}),
                  std::out_of_range);
}
