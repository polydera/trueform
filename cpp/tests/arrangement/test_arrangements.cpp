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
#include "mixed_mesh.hpp"
#include "trueform/cpp/arrangement/async/mesh_arrangements.hpp"
#include "trueform/cpp/arrangement/async/polygon_arrangements.hpp"
#include "trueform/cpp/arrangement/mesh_arrangements.hpp"
#include "trueform/cpp/arrangement/polygon_arrangements.hpp"
#include "trueform/cpp/clean/mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/positively_oriented.hpp"
#include "trueform/cpp/intersect/async/intersection_curves.hpp"
#include "trueform/cpp/intersect/async/self_intersection_curves.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"
#include "trueform/cpp/topology/boundary_paths.hpp"
#include "trueform/cpp/topology/connect_edges_to_paths.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"
#include "trueform/cpp/topology/non_manifold_edges.hpp"
#include "trueform/cpp/topology/orient_faces_consistently.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

/// A placement is the caller's own sixteen numbers, which is what the assembly
/// takes: it is copied into the owner, so nothing here shares one.
template <typename Real>
auto translation(Real x, Real y = Real{0}, Real z = Real{0})
    -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1}, Real{0}, y,
          Real{0}, Real{0}, Real{1}, z, Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
using owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real> auto horizontal_triangle() -> owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0},
                  Real{0}, Real{1}, Real{0}})};
}

template <typename Real>
auto vertical_triangle(bool transformed = false) -> owned<Real> {
  const auto offset = transformed ? Real{7} : Real{0};
  owned<Real> result{tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {offset, Real{-0.5}, Real{-1}, offset, Real{-0.5}, Real{1},
                  offset, Real{0.75}, Real{0}})};
  if (transformed)
    result.place(translation<Real>(-offset));
  return result;
}

template <typename Real> auto self_crossing_mesh() -> owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{0},
       Real{1}, Real{0}, Real{0}, Real{-0.5}, Real{-1}, Real{0}, Real{-0.5},
       Real{1}, Real{0}, Real{0.75}, Real{0}})};
}

template <typename Real> auto empty_mesh() -> owned<Real> { return {}; }

/// The views a range entry takes, over owners the caller keeps: the vector is
/// final before the first view is taken of it.
template <typename Owned>
auto meshes_of(const std::vector<Owned> &owners)
    -> std::vector<typename Owned::mesh_type> {
  std::vector<typename Owned::mesh_type> meshes;
  meshes.reserve(owners.size());
  for (const auto &owner : owners)
    meshes.push_back(owner.mesh());
  return meshes;
}

/// A result is core's own storage, so two of them are compared where they lie.
template <typename Polygons>
auto same_points(const Polygons &a, const Polygons &b) -> bool {
  const auto &first = a.points_buffer().data_buffer();
  const auto &second = b.points_buffer().data_buffer();
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin());
}

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::mesh_arrangement_result<tf::cpp::default_index_t,
                                                     Real, Ngon> &result)
    -> bool {
  return result.tag_labels.is_valid() && result.face_labels.is_valid() &&
         result.tag_labels.ndim() == 1 && result.face_labels.ndim() == 1 &&
         result.tag_labels.length() == result.mesh.size() &&
         result.face_labels.length() == result.tag_labels.length();
}

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::mesh_arrangement_with_curves_result<
              tf::cpp::default_index_t, Real, Ngon> &result) -> bool {
  return result.tag_labels.is_valid() && result.face_labels.is_valid() &&
         result.tag_labels.length() == result.mesh.size() &&
         result.face_labels.length() == result.tag_labels.length();
}

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::polygon_arrangement_result<
              tf::cpp::default_index_t, Real, Ngon> &result) -> bool {
  return result.face_labels.is_valid() && result.face_labels.ndim() == 1 &&
         result.face_labels.length() == result.mesh.size();
}

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::polygon_arrangement_with_curves_result<
              tf::cpp::default_index_t, Real, Ngon> &result) -> bool {
  return result.face_labels.is_valid() &&
         result.face_labels.length() == result.mesh.size();
}

struct counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_TEST_CASE("mesh arrangements expose labels curves caches and modes",
                   "[cpp][arrangement][arrangements][mesh]", float, double) {
  const std::vector<owned<TestType>> owners{horizontal_triangle<TestType>(),
                                            vertical_triangle<TestType>()};
  const auto meshes = meshes_of(owners);

  const auto plain =
      tf::cpp::mesh_arrangements(meshes, {tf::intersect_mode::sos, 0.0});
  CHECK(coherent(plain));
  CHECK(plain.mesh.size() >= 2);
  // a mesh is a reading of the caller's own geometry through the caller's own
  // cache, so what the pipeline warmed is what the caller holds
  CHECK(owners[0].cache.is_tree_fresh(meshes[0].geometry()));
  CHECK(owners[0].cache.is_face_membership_fresh(meshes[0].geometry()));
  CHECK(owners[0].cache.is_manifold_edge_link_fresh(meshes[0].geometry()));

  const auto with_curves = tf::cpp::mesh_arrangements_with_curves(
      meshes, {{tf::intersect_mode::primitives |
                    tf::intersect_mode::resolve_crossing_contours,
                1e-6},
               tf::triangulation_type::refined_cdt});
  CHECK(coherent(with_curves));
  CHECK(with_curves.curves.size() > 0);
  CHECK(with_curves.mesh.size() >= 2);
}

TEMPLATE_TEST_CASE("polygon arrangements expose labels curves and local frame",
                   "[cpp][arrangement][arrangements][polygon]", float, double) {
  const auto crossing = self_crossing_mesh<TestType>();
  const auto mesh = crossing.mesh();
  const auto plain = tf::cpp::polygon_arrangements(mesh);
  const auto with_curves = tf::cpp::polygon_arrangements_with_curves(mesh);

  CHECK(coherent(plain));
  CHECK(coherent(with_curves));
  CHECK(plain.mesh.size() > crossing.polygons.size());
  CHECK(with_curves.curves.size() > 0);
  CHECK(crossing.cache.is_tree_fresh(mesh.geometry()));
  CHECK(crossing.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(crossing.cache.is_manifold_edge_link_fresh(mesh.geometry()));

  auto transformed = self_crossing_mesh<TestType>();
  transformed.place(translation<TestType>(9, -3, 2));
  const auto transformed_result =
      tf::cpp::polygon_arrangements(transformed.mesh());
  CHECK(same_points(plain.mesh, transformed_result.mesh));
}

TEMPLATE_TEST_CASE("mesh arrangements apply mixed stored transformations",
                   "[cpp][arrangement][arrangements][transform]", float,
                   double) {
  const std::vector<owned<TestType>> owners{horizontal_triangle<TestType>(),
                                            vertical_triangle<TestType>(true)};
  const auto meshes = meshes_of(owners);

  const auto result = tf::cpp::mesh_arrangements_with_curves(meshes);
  CHECK(coherent(result));
  CHECK(result.curves.size() > 0);
  // the placement moved the coordinates rather than travelling with them
  for (const auto point : result.mesh.points())
    CHECK(point[0] < TestType{2});
}

TEMPLATE_TEST_CASE("arrangements return coherent empty outputs",
                   "[cpp][arrangement][arrangements][empty]", float, double) {
  const std::vector<owned<TestType>> owners{empty_mesh<TestType>(),
                                            empty_mesh<TestType>()};
  const auto meshes = meshes_of(owners);

  const auto mesh_result = tf::cpp::mesh_arrangements(meshes);
  const auto mesh_curves = tf::cpp::mesh_arrangements_with_curves(meshes);
  const auto polygon_result = tf::cpp::polygon_arrangements(meshes[0]);
  const auto polygon_curves =
      tf::cpp::polygon_arrangements_with_curves(meshes[1]);

  CHECK(coherent(mesh_result));
  CHECK(mesh_result.mesh.size() == 0);
  CHECK(mesh_result.mesh.points_buffer().size() == 0);
  CHECK(coherent(mesh_curves));
  CHECK(mesh_curves.curves.size() == 0);
  CHECK(coherent(polygon_result));
  CHECK(polygon_result.mesh.size() == 0);
  CHECK(coherent(polygon_curves));
  CHECK(polygon_curves.curves.size() == 0);
}

TEMPLATE_TEST_CASE("arrangements validate lists handles indices and configs",
                   "[cpp][arrangement][arrangements][validation]", float,
                   double) {
  const auto empty_owner = empty_mesh<TestType>();
  const auto empty = empty_owner.mesh();
  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>>
      none;
  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>>
      one{empty};
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(none), std::runtime_error);
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(one), std::runtime_error);

  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>>
      empties{empty, empty};
  CHECK(tf::cpp::mesh_arrangements(empties).mesh.size() == 0);
  CHECK(tf::cpp::polygon_arrangements(empty).mesh.size() == 0);

  // faces that name a point the geometry does not have are refused where the
  // reading is answered for, which is the one door they pass
  const owned<TestType> malformed_owner{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  const auto malformed = malformed_owner.mesh();
  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>>
      malformed_meshes{empty, malformed};
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(malformed_meshes),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::polygon_arrangements(malformed), std::out_of_range);

  const auto check_bad_tolerance = [&](double tolerance) {
    CHECK_THROWS_AS(tf::cpp::mesh_arrangements(
                        empties, {tf::intersect_mode::primitives, tolerance}),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::polygon_arrangements(
                        empty, {tf::intersect_mode::primitives, tolerance}),
                    std::invalid_argument);
  };
  check_bad_tolerance(-1.0);
  check_bad_tolerance(std::numeric_limits<double>::infinity());
  check_bad_tolerance(std::numeric_limits<double>::quiet_NaN());
  if constexpr (std::is_same_v<TestType, float>)
    check_bad_tolerance(std::numeric_limits<double>::max());

  for (const auto mode :
       {0,
        static_cast<int>(tf::intersect_mode::sos) |
            static_cast<int>(tf::intersect_mode::primitives),
        static_cast<int>(tf::intersect_mode::primitives) | 32}) {
    const auto config =
        tf::intersect_config{static_cast<tf::intersect_mode>(mode), 0.0};
    CHECK_THROWS_AS(tf::cpp::mesh_arrangements(empties, config),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::polygon_arrangements(empty, config),
                    std::invalid_argument);
  }

  CHECK(coherent(tf::cpp::mesh_arrangements(
      empties,
      {tf::intersect_mode::primitives | tf::intersect_mode::within, 0.0})));
}

TEMPLATE_TEST_CASE("arrangement async entries read the operands they borrow",
                   "[cpp][arrangement][arrangements][async]", float, double) {
  const std::vector<owned<TestType>> owners{horizontal_triangle<TestType>(),
                                            vertical_triangle<TestType>()};
  const auto crossing_owner = self_crossing_mesh<TestType>();
  const auto meshes = meshes_of(owners);
  const auto crossing = crossing_owner.mesh();

  auto mesh_future = tf::cpp::async::mesh_arrangements(meshes);
  auto mesh_curves_future =
      tf::cpp::async::mesh_arrangements_with_curves(meshes);
  auto polygon_future = tf::cpp::async::polygon_arrangements(crossing);
  auto polygon_curves_future =
      tf::cpp::async::polygon_arrangements_with_curves(crossing);
  static_assert(std::is_same_v<decltype(mesh_future),
                               std::future<tf::cpp::mesh_arrangement_result<
                                   tf::cpp::default_index_t, TestType, 3>>>);
  static_assert(
      std::is_same_v<decltype(mesh_curves_future),
                     std::future<tf::cpp::mesh_arrangement_with_curves_result<
                         tf::cpp::default_index_t, TestType, 3>>>);
  static_assert(std::is_same_v<decltype(polygon_future),
                               std::future<tf::cpp::polygon_arrangement_result<
                                   tf::cpp::default_index_t, TestType, 3>>>);
  static_assert(std::is_same_v<
                decltype(polygon_curves_future),
                std::future<tf::cpp::polygon_arrangement_with_curves_result<
                    tf::cpp::default_index_t, TestType, 3>>>);

  CHECK(coherent(mesh_future.get()));
  CHECK(coherent(mesh_curves_future.get()));
  CHECK(coherent(polygon_future.get()));
  CHECK(coherent(polygon_curves_future.get()));

  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto custom_crossing = self_crossing_mesh<TestType>();
  auto custom_future = tf::cpp::async::polygon_arrangements_with_curves(
      counting_resolver{submissions}, custom_crossing.mesh());
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom_future.get().curves.size() > 0);

  // a caller whose own handle cannot outlive the call assembles with the
  // keepalive the constructor takes, and the job reads through it
  std::shared_ptr<const owned<TestType>> held =
      std::make_shared<const owned<TestType>>(self_crossing_mesh<TestType>());
  const tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3> retained(
      held->polygons.faces(), held->polygons.points(), held->cache,
      held->frame(), held);
  auto retained_future = tf::cpp::async::polygon_arrangements(retained);
  held = {};
  CHECK(coherent(retained_future.get()));

  const auto empty_owner = empty_mesh<TestType>();
  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>>
      empties{empty_owner.mesh(), empty_owner.mesh()};
  auto failed = tf::cpp::async::mesh_arrangements(
      empties, {tf::intersect_mode::primitives,
                std::numeric_limits<double>::infinity()});
  CHECK_THROWS_AS(failed.get(), std::invalid_argument);
}

namespace {

template <typename Index, typename Real> struct arrangement_matrix_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using result_mesh_type = tf::polygons_buffer<Index, Real, 3, 3>;
  using curves_type = tf::curves_buffer<Index, Real, 3>;
};

using arrangement_float_int32 = arrangement_matrix_row<std::int32_t, float>;
using arrangement_float_int64 = arrangement_matrix_row<std::int64_t, float>;
using arrangement_double_int32 = arrangement_matrix_row<std::int32_t, double>;
using arrangement_double_int64 = arrangement_matrix_row<std::int64_t, double>;

template <typename Row, std::size_t Ngon>
auto typed_empty_mesh()
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  return {};
}

/// One triangle, stated at whichever arity the row asks for: the layout is the
/// geometry's own, so the offsets are what a mixed mesh states beside it.
template <typename Row, std::size_t Ngon, typename Coordinates>
auto typed_one_triangle(const Coordinates &points)
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  if constexpr (Ngon == 3)
    return {tf::cpp::test::polygons_of<Index, Real>({0, 1, 2}, points)};
  else
    return {tf::cpp::test::polygons_of<Index, Real>({0, 3}, {0, 1, 2}, points)};
}

template <typename Row, std::size_t Ngon>
auto typed_horizontal(bool transformed = false)
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  const auto offset = transformed ? Real{7} : Real{0};
  const std::initializer_list<Real> points{offset - Real{1}, Real{-1}, Real{0},
                                           offset + Real{1}, Real{-1}, Real{0},
                                           offset,           Real{1},  Real{0}};
  auto result = typed_one_triangle<Row, Ngon>(points);
  if (transformed)
    result.place(translation<Real>(-offset));
  return result;
}

template <typename Row, std::size_t Ngon>
auto typed_vertical() -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  const std::initializer_list<Real> points{Real{0}, Real{-0.5}, Real{-1},
                                           Real{0}, Real{-0.5}, Real{1},
                                           Real{0}, Real{0.75}, Real{0}};
  return typed_one_triangle<Row, Ngon>(points);
}

template <typename Row, std::size_t Ngon>
auto typed_crossing() -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const std::initializer_list<Real> points{
      Real{-1}, Real{-1},   Real{0}, Real{1}, Real{-1},   Real{0},
      Real{0},  Real{1},    Real{0}, Real{0}, Real{-0.5}, Real{-1},
      Real{0},  Real{-0.5}, Real{1}, Real{0}, Real{0.75}, Real{0}};
  if constexpr (Ngon == 3)
    return {
        tf::cpp::test::polygons_of<Index, Real>({0, 1, 2, 3, 4, 5}, points)};
  else
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 3, 6}, {0, 1, 2, 3, 4, 5}, points)};
}

struct arrangement_curve_statistics {
  std::size_t paths = 0;
  std::vector<std::size_t> edge_counts;
  std::size_t closed = 0;
  std::size_t open = 0;
  std::size_t endpoints = 0;

  friend auto operator==(const arrangement_curve_statistics &a,
                         const arrangement_curve_statistics &b) -> bool {
    return a.paths == b.paths && a.edge_counts == b.edge_counts &&
           a.closed == b.closed && a.open == b.open &&
           a.endpoints == b.endpoints;
  }
};

/// The same statistics, whichever carrier states the paths: an offset-blocked
/// buffer a caller holds, or the two arrays a curve result is.
template <typename Index>
auto path_statistics(const tf::cpp::nd_array<Index> &offsets,
                     const tf::cpp::nd_array<Index> &data, std::size_t count)
    -> arrangement_curve_statistics {
  arrangement_curve_statistics result;
  result.paths = count;
  std::set<Index> endpoints;
  for (std::size_t path = 0; path + 1 < offsets.length(); ++path) {
    const auto begin = offsets[path];
    const auto end = offsets[path + 1];
    const auto count = static_cast<std::size_t>(end - begin);
    result.edge_counts.push_back(count == 0 ? 0 : count - 1);
    if (count >= 2 && data[static_cast<std::size_t>(begin)] ==
                          data[static_cast<std::size_t>(end - 1)]) {
      ++result.closed;
    } else {
      ++result.open;
      if (count >= 2) {
        endpoints.insert(data[static_cast<std::size_t>(begin)]);
        endpoints.insert(data[static_cast<std::size_t>(end - 1)]);
      }
    }
  }
  std::sort(result.edge_counts.begin(), result.edge_counts.end());
  result.endpoints = endpoints.size();
  return result;
}

template <typename Index>
auto path_statistics(const tf::cpp::offset_blocked_buffer<Index, Index> &paths)
    -> arrangement_curve_statistics {
  return path_statistics(paths.offsets(), paths.data(),
                         static_cast<std::size_t>(paths.size()));
}

template <typename Index, typename Real>
auto curve_statistics(const tf::curves_buffer<Index, Real, 3> &curves)
    -> arrangement_curve_statistics {
  const auto arrays = tf::cpp::test::arrays_of(curves);
  return path_statistics(arrays.offsets, arrays.ids,
                         static_cast<std::size_t>(curves.size()));
}

template <typename Curves0, typename Curves1>
auto same_curve_points(const Curves0 &a, const Curves1 &b) -> bool {
  using Real = std::decay_t<decltype(a.points()[0][0])>;
  auto gather = [](const auto &curves) {
    std::vector<std::array<Real, 3>> points;
    for (const auto point : curves.points())
      points.push_back({point[0], point[1], point[2]});
    std::sort(points.begin(), points.end());
    return points;
  };
  return gather(a) == gather(b);
}

// An uncut face is emitted verbatim, so the result states the arity the
// operands did and a mixed operand's arrangement is a mixed mesh.
template <typename Result>
auto typed_result_is_coherent(const Result &result) -> bool {
  if (!result.tag_labels.is_valid() || !result.face_labels.is_valid() ||
      result.tag_labels.ndim() != 1 || result.face_labels.ndim() != 1 ||
      result.tag_labels.length() != result.face_labels.length() ||
      result.tag_labels.length() != result.mesh.size())
    return false;
  const auto indices = tf::cpp::test::face_indices_of(result.mesh);
  const auto points =
      static_cast<std::int64_t>(result.mesh.points_buffer().size());
  for (const auto point : indices.make_range())
    if (point < 0 || static_cast<std::int64_t>(point) >= points)
      return false;
  return true;
}

template <typename Result>
auto typed_polygon_result_is_coherent(const Result &result) -> bool {
  return result.face_labels.is_valid() && result.face_labels.ndim() == 1 &&
         result.face_labels.length() == result.mesh.size();
}

template <typename Index, typename Real>
auto python_sphere(Real center_x, Real center_y)
    -> tf::cpp::test::owned_mesh<Index, Real> {
  const tf::cpp::test::owned_mesh<Index, Real> sphere{
      tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 40, 40)};
  tf::cpp::test::owned_mesh<Index, Real> oriented{
      tf::cpp::positively_oriented(sphere.mesh())};
  for (auto point : oriented.polygons.points()) {
    point[0] += center_x;
    point[1] += center_y;
  }
  oriented.cache.points_changed();
  return oriented;
}

template <typename Row, std::size_t Ngon>
auto check_arrangement_case() -> void {
  using Index = typename Row::index_type;
  using Owned = tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon>;
  const std::vector<Owned> owners{typed_horizontal<Row, Ngon>(true),
                                  typed_vertical<Row, Ngon>()};
  const auto crossing_owner = typed_crossing<Row, Ngon>();
  const auto forms = meshes_of(owners);
  const auto crossing = crossing_owner.mesh();

  const auto config =
      tf::arrangement_config{{tf::intersect_mode::primitives |
                                  tf::intersect_mode::resolve_crossing_contours,
                              0.0},
                             tf::triangulation_type::cdt};
  const auto plain = tf::cpp::mesh_arrangements(forms, config);
  const auto with_curves =
      tf::cpp::mesh_arrangements_with_curves(forms, config);
  const auto polygon = tf::cpp::polygon_arrangements(crossing, config);
  const auto polygon_curves =
      tf::cpp::polygon_arrangements_with_curves(crossing, config);
  REQUIRE(typed_result_is_coherent(plain));
  REQUIRE(typed_result_is_coherent(with_curves));
  REQUIRE(typed_polygon_result_is_coherent(polygon));
  REQUIRE(typed_polygon_result_is_coherent(polygon_curves));
  CHECK(plain.mesh.size() >= 2);
  CHECK(polygon.mesh.size() > crossing_owner.polygons.size());

  std::set<Index> tags;
  for (std::size_t face = 0; face < plain.tag_labels.length(); ++face) {
    tags.insert(plain.tag_labels[face]);
    CHECK(plain.face_labels[face] == Index{0});
  }
  CHECK(tags == std::set<Index>{Index{0}, Index{1}});

  const auto intersection =
      tf::cpp::intersection_curves(forms, config.intersect);
  const auto self =
      tf::cpp::self_intersection_curves(crossing, config.intersect);
  const auto expected_statistics = curve_statistics(intersection);
  CHECK(expected_statistics == curve_statistics(self));
  CHECK(expected_statistics == curve_statistics(with_curves.curves));
  CHECK(expected_statistics == curve_statistics(polygon_curves.curves));
  CHECK(same_curve_points(intersection, self));
  CHECK(same_curve_points(intersection, with_curves.curves));
  CHECK(same_curve_points(intersection, polygon_curves.curves));

  const Owned topology{with_curves.mesh};
  const auto non_manifold = tf::cpp::non_manifold_edges(topology.mesh());
  const auto non_manifold_paths = tf::cpp::connect_edges_to_paths(non_manifold);
  CHECK(expected_statistics == path_statistics(non_manifold_paths));

  const auto refined = tf::cpp::mesh_arrangements(
      forms, {config.intersect, tf::triangulation_type::refined_cdt});
  CHECK(refined.mesh.points_buffer().size() >=
        plain.mesh.points_buffer().size());
  CHECK(refined.mesh.size() >= plain.mesh.size());
  // a mesh is a reading of the caller's own geometry through the caller's own
  // cache, so what the pipeline warmed is what the caller holds
  CHECK(owners[0].cache.is_tree_fresh(forms[0].geometry()));
  CHECK(owners[1].cache.is_tree_fresh(forms[1].geometry()));
  CHECK(owners[0].cache.is_manifold_edge_link_fresh(forms[0].geometry()));
  CHECK(owners[1].cache.is_manifold_edge_link_fresh(forms[1].geometry()));
}

} // namespace

TEMPLATE_TEST_CASE(
    "arrangements cover the Python real index and every operand arity",
    "[cpp][arrangement][arrangements][matrix][python-parity]",
    arrangement_float_int32, arrangement_float_int64, arrangement_double_int32,
    arrangement_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using ResultMesh = typename TestType::result_mesh_type;
  using Curves = typename TestType::curves_type;
  using MeshResult = tf::cpp::mesh_arrangement_result<Index, Real, 3>;
  using MeshCurvesResult =
      tf::cpp::mesh_arrangement_with_curves_result<Index, Real, 3>;
  using PolygonResult = tf::cpp::polygon_arrangement_result<Index, Real, 3>;
  using PolygonCurvesResult =
      tf::cpp::polygon_arrangement_with_curves_result<Index, Real, 3>;

  static_assert(
      std::is_same_v<decltype(std::declval<MeshResult>().mesh), ResultMesh>);
  static_assert(std::is_same_v<decltype(std::declval<MeshResult>().tag_labels),
                               tf::cpp::nd_array<Index>>);
  static_assert(
      std::is_same_v<decltype(std::declval<MeshCurvesResult>().curves),
                     Curves>);
  static_assert(
      std::is_same_v<decltype(std::declval<PolygonResult>().face_labels),
                     tf::cpp::nd_array<Index>>);
  static_assert(
      std::is_same_v<decltype(std::declval<PolygonCurvesResult>().curves),
                     Curves>);

  check_arrangement_case<TestType, 3>();
  check_arrangement_case<TestType, tf::dynamic_size>();
}

TEST_CASE("a range of views is arranged at the arity its element states",
          "[cpp][arrangement][arrangements][matrix][layout]") {
  using Row = arrangement_float_int32;
  using Mesh = tf::cpp::mesh<std::int32_t, float, 3, 3>;
  using DynamicMesh = tf::cpp::mesh<std::int32_t, float, 3, tf::dynamic_size>;

  // A range is homogeneous -- otherwise it is not a range -- so the element
  // type carries the arity the operands are read at, and the cut surface is
  // triangles either way.
  auto first = typed_horizontal<Row, 3>();
  auto second = typed_vertical<Row, 3>();
  const std::vector<Mesh> triangles{first.mesh(), second.mesh()};
  REQUIRE(typed_result_is_coherent(tf::cpp::mesh_arrangements(triangles)));

  auto mixed_first = typed_horizontal<Row, tf::dynamic_size>();
  auto mixed_second = typed_vertical<Row, tf::dynamic_size>();
  const std::vector<DynamicMesh> mixed{mixed_first.mesh(), mixed_second.mesh()};
  REQUIRE(typed_result_is_coherent(tf::cpp::mesh_arrangements(mixed)));
}

TEST_CASE("typed arrangement async fronts answer for the reading they hold",
          "[cpp][arrangement][arrangements][matrix][async]") {
  using Row = arrangement_double_int64;
  using Owned = tf::cpp::test::mixed_mesh_of<typename Row::mesh_type>;
  using Result =
      tf::cpp::mesh_arrangement_with_curves_result<std::int64_t, double,
                                                   tf::dynamic_size>;
  const std::vector<Owned> owners{typed_horizontal<Row, tf::dynamic_size>(true),
                                  typed_vertical<Row, tf::dynamic_size>()};
  const auto crossing_owner = typed_crossing<Row, tf::dynamic_size>();
  const auto forms = meshes_of(owners);
  const auto crossing = crossing_owner.mesh();
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  auto mesh_pending = tf::cpp::async::mesh_arrangements_with_curves(
      counting_resolver{submissions}, forms);
  static_assert(std::is_same_v<decltype(mesh_pending), std::future<Result>>);
  const auto mesh_result = mesh_pending.get();
  CHECK(mesh_result.curves.size() > 0);

  auto polygon_pending = tf::cpp::async::polygon_arrangements_with_curves(
      counting_resolver{submissions}, crossing);
  static_assert(std::is_same_v<
                decltype(polygon_pending),
                std::future<tf::cpp::polygon_arrangement_with_curves_result<
                    std::int64_t, double, tf::dynamic_size>>>);
  const auto polygon_result = polygon_pending.get();
  CHECK(polygon_result.curves.size() > 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 2);

  const Owned nothing;
  CHECK(
      tf::cpp::async::polygon_arrangements(nothing.mesh()).get().mesh.size() ==
      0);
}

TEMPLATE_TEST_CASE(
    "typed arrangements validate every index carrier and triangulation",
    "[cpp][arrangement][arrangements][matrix][validation]",
    arrangement_float_int32, arrangement_float_int64, arrangement_double_int32,
    arrangement_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Owned = typename TestType::mesh_type;
  using MixedOwned = tf::cpp::test::mixed_mesh_of<Owned>;

  // a default-constructed owner is the EMPTY mesh, which arranges to nothing
  const Owned nothing;
  const auto valid = typed_vertical<TestType, 3>();
  const std::vector<Owned> with_nothing{valid, nothing};
  CHECK(tf::cpp::mesh_arrangements(meshes_of(with_nothing)).mesh.size() ==
        valid.polygons.size());
  CHECK(tf::cpp::polygon_arrangements(nothing.mesh()).mesh.size() == 0);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes, at either arity
  const Owned bad_fixed{tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 3}, {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0},
                  Real{1}, Real{0}})};
  CHECK_THROWS_AS(tf::cpp::polygon_arrangements(bad_fixed.mesh()),
                  std::out_of_range);

  const MixedOwned bad_dynamic{tf::cpp::test::polygons_of<Index, Real>(
      {0, 3}, {0, 1, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}})};
  const std::vector<MixedOwned> bad_forms{
      typed_vertical<TestType, tf::dynamic_size>(), bad_dynamic};
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(meshes_of(bad_forms)),
                  std::out_of_range);

  const auto empty_a = typed_empty_mesh<TestType, 3>();
  const auto empty_b = typed_empty_mesh<TestType, tf::dynamic_size>();
  const std::vector<Owned> empty_owners{empty_a, empty_a};
  const std::vector<MixedOwned> empty_mixed_owners{empty_b, empty_b};
  const auto empty_forms = meshes_of(empty_owners);
  const auto empty_mixed_forms = meshes_of(empty_mixed_owners);
  const auto bad_triangulation =
      tf::arrangement_config{{tf::intersect_mode::primitives, 0.0},
                             static_cast<tf::triangulation_type>(99)};
  CHECK_THROWS_AS(tf::cpp::mesh_arrangements(empty_forms, bad_triangulation),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::polygon_arrangements(empty_a.mesh(), bad_triangulation),
      std::invalid_argument);

  CHECK(typed_result_is_coherent(tf::cpp::mesh_arrangements(empty_forms)));
  CHECK(typed_result_is_coherent(
      tf::cpp::mesh_arrangements_with_curves(empty_mixed_forms)));
  CHECK(typed_polygon_result_is_coherent(
      tf::cpp::polygon_arrangements(empty_b.mesh())));
  CHECK(typed_polygon_result_is_coherent(
      tf::cpp::polygon_arrangements_with_curves(empty_b.mesh())));
  CHECK(typed_polygon_result_is_coherent(
      tf::cpp::polygon_arrangements(empty_a.mesh())));

  if constexpr (std::is_same_v<Index, std::int64_t>) {
    // an identity past int32 names a point the mesh does not have; the
    // arrangement runs at the caller's index and rejects it as out of range
    const auto beyond_int32 =
        static_cast<Index>(std::numeric_limits<std::int32_t>::max()) + Index{1};
    const Owned beyond{tf::cpp::test::polygons_of<Index, Real>(
        {0, 1, beyond_int32}, {Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
                               Real{0}, Real{0}, Real{1}, Real{0}})};
    CHECK_THROWS_AS(tf::cpp::polygon_arrangements(beyond.mesh()),
                    std::out_of_range);
  }
}

TEST_CASE("exact Python three-sphere arrangement fixture preserves topology",
          "[cpp][arrangement][arrangements][python-parity][sphere]") {
  using Real = double;
  using Index = std::int32_t;
  using Owned = tf::cpp::test::owned_mesh<Index, Real>;
  const auto sqrt_three = std::sqrt(Real{3});
  const std::vector<Owned> owners{
      python_sphere<Index, Real>(Real{0}, Real{0}),
      python_sphere<Index, Real>(Real{1}, Real{0}),
      python_sphere<Index, Real>(Real{0.5}, sqrt_three / Real{2})};
  const auto spheres = meshes_of(owners);

  const auto cdt = tf::cpp::mesh_arrangements_with_curves(spheres);
  REQUIRE(typed_result_is_coherent(cdt));
  std::set<Index> tags(cdt.tag_labels.begin(), cdt.tag_labels.end());
  CHECK(tags == std::set<Index>{0, 1, 2});
  for (std::size_t face = 0; face < cdt.face_labels.length(); ++face) {
    const auto tag = cdt.tag_labels[face];
    REQUIRE(tag >= 0);
    REQUIRE(tag < 3);
    CHECK(cdt.face_labels[face] >= 0);
    CHECK(static_cast<std::size_t>(cdt.face_labels[face]) <
          owners[static_cast<std::size_t>(tag)].polygons.size());
  }

  const Owned topology{cdt.mesh};
  CHECK(tf::cpp::boundary_paths(topology.mesh()).size() == 0);
  const auto non_manifold_paths = tf::cpp::connect_edges_to_paths(
      tf::cpp::non_manifold_edges(topology.mesh()));
  const auto statistics = curve_statistics(cdt.curves);
  CHECK(statistics == path_statistics(non_manifold_paths));
  CHECK(statistics.paths == 6);
  CHECK(statistics.endpoints == 2);

  const auto refined = tf::cpp::mesh_arrangements(
      spheres, {{tf::intersect_mode::primitives |
                     tf::intersect_mode::resolve_crossing_contours,
                 0.0},
                tf::triangulation_type::refined_cdt});
  CHECK(refined.mesh.points_buffer().size() >= cdt.mesh.points_buffer().size());
  CHECK(refined.mesh.size() >= cdt.mesh.size());
  const Owned refined_topology{refined.mesh};
  CHECK(tf::cpp::boundary_paths(refined_topology.mesh()).size() == 0);
}

TEST_CASE("arrangement facade never welds one form's rim onto another's wall",
          "[cpp][arrangement][arrangements][python-parity][tolerance]") {
  using Real = float;
  using Index = std::int64_t;
  using Owned = tf::cpp::test::owned_mesh<Index, Real>;
  const auto gap = Real{1e-4};
  const auto half_extent = Real{0.5} - gap;

  const std::vector<Owned> owners{
      Owned{tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1})},
      Owned{tf::cpp::test::polygons_of<Index, Real>(
          {0, 1, 2, 0, 2, 3}, {-half_extent, -half_extent, Real{0}, half_extent,
                               -half_extent, Real{0}, half_extent, half_extent,
                               Real{0}, -half_extent, half_extent, Real{0}})}};
  const auto forms = meshes_of(owners);

  const auto snapped =
      tf::cpp::mesh_arrangements(forms, {tf::intersect_mode::primitives, 1e-3});
  const Owned snapped_owner{snapped.mesh};
  const Owned cleaned{tf::cpp::cleaned_mesh(snapped_owner.mesh(), Real{1e-3})};
  const Owned oriented{tf::cpp::orient_faces_consistently(cleaned.mesh())};
  const auto snapped_labels = tf::cpp::make_domain_labels(
      oriented.mesh(), tf::domain_config::ignore_open_fragments);
  // the band is a licence to move a vertex onto a plane its OWN faces state,
  // so a gap ten times smaller than it still separates two forms
  CHECK(snapped_labels.number_of_domains() == Index{2});

  const auto exact =
      tf::cpp::mesh_arrangements(forms, {tf::intersect_mode::primitives, 0.0});
  const Owned exact_owner{exact.mesh};
  const Owned exact_oriented{
      tf::cpp::orient_faces_consistently(exact_owner.mesh())};
  const auto exact_labels = tf::cpp::make_domain_labels(
      exact_oriented.mesh(), tf::domain_config::ignore_open_fragments);
  CHECK(exact_labels.number_of_domains() == Index{2});
}
