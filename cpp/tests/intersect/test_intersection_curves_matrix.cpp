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
#include "trueform/cpp/core/build_manifold_edge_link.hpp"
#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/intersect/async/intersection_curves.hpp"
#include "trueform/cpp/intersect/async/self_intersection_curves.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Index, typename Real> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using curves_type = tf::curves_buffer<Index, Real, 3>;
};
using float_int32 = matrix_row<std::int32_t, float>;
using float_int64 = matrix_row<std::int64_t, float>;
using double_int32 = matrix_row<std::int32_t, double>;
using double_int64 = matrix_row<std::int64_t, double>;

template <typename A, typename B, typename = void>
struct has_intersection_curves : std::false_type {};
template <typename A, typename B>
struct has_intersection_curves<
    A, B,
    std::void_t<decltype(tf::cpp::intersection_curves(
        std::declval<A &>(), std::declval<B &>()))>> : std::true_type {};
template <typename Mesh, typename = void>
struct has_self_intersection_curves : std::false_type {};
template <typename Mesh>
struct has_self_intersection_curves<
    Mesh, std::void_t<decltype(tf::cpp::self_intersection_curves(
              std::declval<Mesh &>()))>> : std::true_type {};

static_assert(
    has_intersection_curves<tf::cpp::mesh<std::int32_t, float, 3, 3>,
                            tf::cpp::mesh<std::int64_t, float, 3, 3>>::value);
static_assert(
    !has_intersection_curves<tf::cpp::mesh<std::int32_t, float, 2, 3>,
                             tf::cpp::mesh<std::int64_t, float, 2, 3>>::value);
static_assert(has_self_intersection_curves<
              tf::cpp::mesh<std::int64_t, double, 3, 3>>::value);
static_assert(!has_self_intersection_curves<
              tf::cpp::mesh<std::int64_t, double, 2, 3>>::value);

/// The sixteen numbers a placement is, which is what an owner is placed by.
template <typename Real>
auto translation(Real x, Real y = Real{0}, Real z = Real{0})
    -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1}, Real{0}, y,
          Real{0}, Real{0}, Real{1}, z, Real{0}, Real{0}, Real{0}, Real{1}};
}

/// One quad, stated as this arity states it: two triangles or one four-sided
/// face over the same four points.
template <typename Row, std::size_t Ngon>
auto make_mesh(std::initializer_list<typename Row::real_type> points)
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  if constexpr (Ngon == 3)
    return {
        tf::cpp::test::polygons_of<Index, Real>({0, 1, 2, 0, 2, 3}, points)};
  else
    return {
        tf::cpp::test::polygons_of<Index, Real>({0, 4}, {0, 1, 2, 3}, points)};
}

template <typename Row, std::size_t Ngon>
auto horizontal() -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  return make_mesh<Row, Ngon>({Real{-1}, Real{-1}, 0, Real{1}, Real{-1}, 0,
                               Real{1}, Real{1}, 0, Real{-1}, Real{1}, 0});
}

template <typename Row, std::size_t Ngon>
auto vertical(typename Row::real_type x = 0)
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  return make_mesh<Row, Ngon>({x, Real{-0.5}, Real{-1}, x, Real{0.5}, Real{-1},
                               x, Real{0.5}, Real{1}, x, Real{-0.5}, Real{1}});
}

template <typename Row, std::size_t Ngon>
auto self_crossing() -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  if constexpr (Ngon != 3)
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 4, 8}, {0, 1, 2, 3, 4, 5, 6, 7},
        {Real{-1}, Real{-1},   Real{0},  Real{1},  Real{-1},   Real{0},
         Real{1},  Real{1},    Real{0},  Real{-1}, Real{1},    Real{0},
         Real{0},  Real{-0.5}, Real{-1}, Real{0},  Real{0.5},  Real{-1},
         Real{0},  Real{0.5},  Real{1},  Real{0},  Real{-0.5}, Real{1}})};
  else
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 1, 2, 3, 4, 5},
        {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{0},
         Real{1}, Real{0}, Real{0}, Real{-0.5}, Real{-1}, Real{0}, Real{-0.5},
         Real{1}, Real{0}, Real{0.75}, Real{0}})};
}

template <typename Row, std::size_t Ngon>
auto empty_mesh() -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  return {};
}

template <typename Curves> auto coherent(const Curves &curves) -> bool {
  const auto arrays = tf::cpp::test::arrays_of(curves);
  if (arrays.points.raw_shape() !=
      tf::small_vector<int, 3>{arrays.points.shape_at(0), 3})
    return false;
  for (const auto point_id : arrays.ids.make_range())
    if (point_id < 0 || point_id >= arrays.points.shape_at(0))
      return false;
  std::size_t iterated = 0;
  for (const auto path : curves.paths()) {
    ++iterated;
    if (path.size() < 2)
      return false;
  }
  return iterated == static_cast<std::size_t>(curves.size());
}

template <typename Curves>
auto exact_pair_segment(const Curves &curves,
                        std::decay_t<decltype(curves.points()[0][0])> x = {})
    -> bool {
  using Real = std::decay_t<decltype(curves.points()[0][0])>;
  if (!coherent(curves) || curves.size() == 0)
    return false;
  auto min_y = std::numeric_limits<Real>::max();
  auto max_y = std::numeric_limits<Real>::lowest();
  for (const auto point : curves.points()) {
    if (point[0] != x || point[2] != Real{0})
      return false;
    min_y = std::min(min_y, point[1]);
    max_y = std::max(max_y, point[1]);
  }
  return min_y == Real{-0.5} && max_y == Real{0.5};
}

template <typename Curves0, typename Curves1>
auto exact_points_equal(const Curves0 &a, const Curves1 &b) -> bool {
  auto gather = [](const auto &curves) {
    using Real = std::decay_t<decltype(curves.points()[0][0])>;
    std::vector<std::array<Real, 3>> points;
    for (const auto point : curves.points())
      points.push_back({point[0], point[1], point[2]});
    std::sort(points.begin(), points.end());
    return points;
  };
  return gather(a) == gather(b);
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

/// The coordinates an owner holds, moved along x where they lie: a test that
/// writes through its own storage then states the change.
template <typename Owned>
auto shift_points(Owned &owner, typename Owned::real_type dx) -> void {
  for (auto point : owner.polygons.points())
    point[0] += dx;
  owner.cache.points_changed();
}

template <typename Row, std::size_t Ngon0, std::size_t Ngon1>
auto check_pair_curves_case() -> void {
  using Index = typename Row::index_type;
  using Curves = typename Row::curves_type;
  const auto owned_a = horizontal<Row, Ngon0>();
  const auto owned_b = vertical<Row, Ngon1>();
  const auto a = owned_a.mesh();
  const auto b = owned_b.mesh();
  const auto *a_points = owned_a.polygons.points_buffer().data_buffer().data();
  const auto *b_points = owned_b.polygons.points_buffer().data_buffer().data();

  auto result = tf::cpp::intersection_curves(a, b);
  static_assert(std::is_same_v<decltype(result), Curves>);
  CHECK(exact_pair_segment(result));
  CHECK(owned_a.polygons.points_buffer().data_buffer().data() == a_points);
  CHECK(owned_b.polygons.points_buffer().data_buffer().data() == b_points);
  // the pipeline runs on the caller's own mesh, so it warms the caller's
  // own structures whatever the index width
  CHECK(owned_a.cache.is_tree_fresh(a.geometry()));
  CHECK(owned_b.cache.is_tree_fresh(b.geometry()));
  CHECK(owned_a.cache.is_face_membership_fresh(a.geometry()));
  CHECK(owned_b.cache.is_face_membership_fresh(b.geometry()));
  CHECK(owned_a.cache.is_manifold_edge_link_fresh(a.geometry()));
  CHECK(owned_b.cache.is_manifold_edge_link_fresh(b.geometry()));

  const auto a_tree_builds = owned_a.cache.tree_build_count();
  const auto b_tree_builds = owned_b.cache.tree_build_count();
  const auto a_membership_builds = owned_a.cache.face_membership_build_count();
  const auto b_membership_builds = owned_b.cache.face_membership_build_count();
  auto repeated = tf::cpp::intersection_curves(a, b);
  CHECK(exact_points_equal(result, repeated));
  CHECK(owned_a.cache.tree_build_count() == a_tree_builds);
  CHECK(owned_b.cache.tree_build_count() == b_tree_builds);
  CHECK(owned_a.cache.face_membership_build_count() == a_membership_builds);
  CHECK(owned_b.cache.face_membership_build_count() == b_membership_builds);

  static_assert(std::is_same_v<
                std::decay_t<decltype(tf::cpp::test::arrays_of(result).ids[0])>,
                Index>);
}

template <typename Row, std::size_t Ngon>
auto check_self_curves_case() -> void {
  using Index = typename Row::index_type;
  const auto hit = self_crossing<Row, Ngon>();
  auto hit_curves = tf::cpp::self_intersection_curves(hit.mesh());
  CHECK(coherent(hit_curves));
  CHECK(hit_curves.size() > 0);
  CHECK(tf::cpp::test::arrays_of(hit_curves).points.shape_at(0) >= 2);
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(tf::cpp::test::arrays_of(hit_curves).ids[0])>,
          Index>);

  const auto miss = horizontal<Row, Ngon>();
  auto miss_curves = tf::cpp::self_intersection_curves(miss.mesh());
  CHECK(coherent(miss_curves));
  CHECK(miss_curves.size() == 0);
  CHECK(tf::cpp::test::arrays_of(miss_curves).points.raw_shape() ==
        tf::small_vector<int, 3>{0, 3});

  const auto empty = empty_mesh<Row, Ngon>();
  auto empty_curves = tf::cpp::self_intersection_curves(empty.mesh());
  CHECK(coherent(empty_curves));
  CHECK(empty_curves.size() == 0);
}

} // namespace

TEMPLATE_TEST_CASE(
    "exact curves cover the Python real index and runtime layout matrix",
    "[cpp][intersect][matrix][python-parity][fixed][dynamic]", float_int32,
    float_int64, double_int32, double_int64) {
  check_pair_curves_case<TestType, 3, 3>();
  check_pair_curves_case<TestType, 3, tf::dynamic_size>();
  check_pair_curves_case<TestType, tf::dynamic_size, 3>();
  check_pair_curves_case<TestType, tf::dynamic_size, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "exact self curves cover hit miss iteration and typed path identities",
    "[cpp][intersect][matrix][python-parity][self]", float_int32, float_int64,
    double_int32, double_int64) {
  check_self_curves_case<TestType, 3>();
  check_self_curves_case<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "exact curves preserve stored transformations symmetry and miss semantics",
    "[cpp][intersect][matrix][python-parity][transform][symmetry]", float_int32,
    float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  auto owned_a = horizontal<TestType, tf::dynamic_size>();
  auto owned_b = vertical<TestType, tf::dynamic_size>();
  owned_a.place(translation<Real>(Real{4}));
  owned_b.place(translation<Real>(Real{4}));
  const auto a = owned_a.mesh();
  const auto b = owned_b.mesh();
  const auto ab = tf::cpp::intersection_curves(a, b);
  const auto ba = tf::cpp::intersection_curves(b, a);
  CHECK(exact_pair_segment(ab, Real{4}));
  CHECK(exact_pair_segment(ba, Real{4}));
  CHECK(exact_points_equal(ab, ba));

  const auto miss = vertical<TestType, tf::dynamic_size>(Real{8});
  const auto no_hit = tf::cpp::intersection_curves(a, miss.mesh());
  CHECK(coherent(no_hit));
  CHECK(no_hit.size() == 0);
}

TEST_CASE("mixed index exact pairs use the lossless common typed carrier",
          "[cpp][intersect][matrix][mixed-index][python-parity]") {
  const auto owned_a32 = horizontal<float_int32, tf::dynamic_size>();
  const auto owned_b64 = vertical<float_int64, 3>();
  const auto a32 = owned_a32.mesh();
  const auto b64 = owned_b64.mesh();
  auto forward = tf::cpp::intersection_curves(a32, b64);
  auto reverse = tf::cpp::intersection_curves(b64, a32);
  static_assert(std::is_same_v<decltype(forward),
                               tf::curves_buffer<std::int64_t, float, 3>>);
  static_assert(std::is_same_v<decltype(reverse),
                               tf::curves_buffer<std::int64_t, float, 3>>);
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(tf::cpp::test::arrays_of(forward).ids[0])>,
          std::int64_t>);
  CHECK(exact_pair_segment(forward));
  CHECK(exact_points_equal(forward, reverse));
}

TEST_CASE("an int64 identity beyond int32 is refused where the reading is",
          "[cpp][intersect][matrix][int64]") {
  // the pipeline runs at the caller's index, so a wide identity is work; what
  // is refused is a face naming a point the geometry does not have
  const auto outside_int32 =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  tf::cpp::test::owned_mesh<std::int64_t, double, 3> wide;
  wide.polygons = tf::cpp::test::polygons_of<std::int64_t, double>(
      {0, 1, outside_int32}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  const auto valid64 = vertical<double_int64, 3>();
  CHECK_THROWS_AS(static_cast<void>(tf::cpp::intersection_curves(
                      wide.mesh(), valid64.mesh())),
                  std::out_of_range);
}

TEST_CASE("an int64 pair answers from the reading it was handed",
          "[cpp][intersect][matrix][int64][async]") {
  const auto owned_a = horizontal<double_int64, 3>();
  auto owned_b = vertical<double_int64, 3>();
  const auto a = owned_a.mesh();

  CHECK(exact_pair_segment(tf::cpp::intersection_curves(a, owned_b.mesh())));
  CHECK(exact_pair_segment(tf::cpp::intersection_curves(a, owned_b.mesh())));
  CHECK(owned_a.cache.is_tree_fresh(a.geometry()));
  CHECK(owned_b.cache.is_tree_fresh(owned_b.mesh().geometry()));

  // a placement is the instance's, so the same geometry read elsewhere misses
  owned_b.place(translation<double>(3.0));
  CHECK(tf::cpp::intersection_curves(a, owned_b.mesh()).size() == 0);
  owned_b.placement.placed = false;
  CHECK(exact_pair_segment(tf::cpp::intersection_curves(a, owned_b.mesh())));

  // the caller states the change, and the reading assembled after it is the
  // one the entry answers for
  shift_points(owned_b, 4.0);
  CHECK(tf::cpp::intersection_curves(a, owned_b.mesh()).size() == 0);
  shift_points(owned_b, -4.0);
  CHECK(exact_pair_segment(tf::cpp::intersection_curves(a, owned_b.mesh())));

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending = tf::cpp::async::intersection_curves(
      counting_resolver{submissions}, a, owned_b.mesh());
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(exact_pair_segment(pending.get()));
}

TEST_CASE("a copy answers from the geometry it was made of",
          "[cpp][intersect][matrix][int64][copy]") {
  const auto a = horizontal<double_int64, 3>();
  auto b = vertical<double_int64, 3>();
  CHECK(exact_pair_segment(tf::cpp::intersection_curves(a.mesh(), b.mesh())));

  // a test's storage is a value, so the copy keeps the geometry it was made
  // of while the original moves out of the way
  const auto copy = b;
  shift_points(b, 4.0);

  CHECK(
      exact_pair_segment(tf::cpp::intersection_curves(a.mesh(), copy.mesh())));
  CHECK(tf::cpp::intersection_curves(a.mesh(), b.mesh()).size() == 0);
}

TEST_CASE("concurrent int64 pairs answer alike",
          "[cpp][intersect][matrix][int64][concurrent]") {
  const auto a = horizontal<double_int64, 3>();
  const auto b = vertical<double_int64, 3>();
  std::vector<std::future<tf::curves_buffer<std::int64_t, double, 3>>> pending;
  // a cache shared across threads is filled before it is shared
  tf::cpp::build_tree(a.mesh());
  tf::cpp::build_tree(b.mesh());
  tf::cpp::build_manifold_edge_link(a.mesh());
  tf::cpp::build_manifold_edge_link(b.mesh());
  for (int task = 0; task < 8; ++task)
    pending.push_back(std::async(std::launch::async, [&a, &b] {
      return tf::cpp::intersection_curves(a.mesh(), b.mesh());
    }));
  for (auto &result : pending)
    CHECK(exact_pair_segment(result.get()));
}

TEST_CASE("a range of meshes answers from the caller's own storage",
          "[cpp][intersect][matrix][list]") {
  const std::vector<tf::cpp::test::owned_mesh<std::int64_t, double, 3>> owners{
      horizontal<double_int64, 3>(), vertical<double_int64, 3>()};
  const std::vector<tf::cpp::mesh<std::int64_t, double, 3, 3>> forms{
      owners[0].mesh(), owners[1].mesh()};
  const auto *a_points =
      owners[0].polygons.points_buffer().data_buffer().data();
  const auto *b_points =
      owners[1].polygons.points_buffer().data_buffer().data();
  auto curves = tf::cpp::intersection_curves(forms);
  CHECK(exact_pair_segment(curves));
  CHECK(owners[0].polygons.points_buffer().data_buffer().data() == a_points);
  CHECK(owners[1].polygons.points_buffer().data_buffer().data() == b_points);
  // the pipeline runs on the caller's forms, so it warms their structures
  CHECK(owners[0].cache.is_tree_fresh(forms[0].geometry()));
  CHECK(owners[1].cache.is_tree_fresh(forms[1].geometry()));
  CHECK(owners[0].cache.is_face_membership_fresh(forms[0].geometry()));
  CHECK(owners[1].cache.is_face_membership_fresh(forms[1].geometry()));
}

TEST_CASE("typed exact async fronts answer for the reading they hold",
          "[cpp][intersect][matrix][async]") {
  const auto a = horizontal<double_int64, tf::dynamic_size>();
  const auto b = vertical<double_int64, 3>();
  const auto crossing = self_crossing<double_int64, tf::dynamic_size>();
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  auto pair_pending = tf::cpp::async::intersection_curves(
      counting_resolver{submissions}, a.mesh(), b.mesh());
  static_assert(
      std::is_same_v<decltype(pair_pending),
                     std::future<tf::curves_buffer<std::int64_t, double, 3>>>);
  CHECK(exact_pair_segment(pair_pending.get()));

  auto self_pending = tf::cpp::async::self_intersection_curves(
      counting_resolver{submissions}, crossing.mesh());
  CHECK(self_pending.get().size() > 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 2);

  const auto legacy_a = horizontal<float_int32, 3>();
  const auto legacy_b = vertical<float_int32, 3>();
  auto legacy_pending = tf::cpp::async::intersection_curves(
      counting_resolver{submissions}, legacy_a.mesh(), legacy_b.mesh());
  static_assert(
      std::is_same_v<
          decltype(legacy_pending),
          std::future<tf::curves_buffer<tf::cpp::default_index_t, float, 3>>>);
  CHECK(exact_pair_segment(legacy_pending.get()));
  CHECK(submissions->load(std::memory_order_relaxed) == 3);

  const tf::cpp::test::owned_mesh<std::int64_t, double, 3> nothing;
  CHECK(tf::cpp::async::self_intersection_curves(nothing.mesh()).get().size() ==
        0);
}

TEMPLATE_TEST_CASE("exact curves validate typed fixed and dynamic indices",
                   "[cpp][intersect][matrix][validation]", float_int32,
                   float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Owned = typename TestType::mesh_type;
  // a default-constructed owner is the EMPTY mesh, which meets nothing
  const Owned nothing;
  const auto valid = vertical<TestType, 3>();
  CHECK(tf::cpp::intersection_curves(nothing.mesh(), valid.mesh()).size() == 0);
  CHECK(tf::cpp::self_intersection_curves(nothing.mesh()).size() == 0);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes, at either arity
  const Owned bad_fixed{tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 4}, {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0},
                  Real{1}, Real{0}})};
  CHECK_THROWS_AS(tf::cpp::intersection_curves(bad_fixed.mesh(), valid.mesh()),
                  std::out_of_range);

  const tf::cpp::test::mixed_mesh_of<Owned> bad_dynamic{
      tf::cpp::test::polygons_of<Index, Real>({0, 3}, {0, 1, 4},
                                              {Real{0}, Real{0}, Real{0},
                                               Real{1}, Real{0}, Real{0},
                                               Real{0}, Real{1}, Real{0}})};
  CHECK_THROWS_AS(tf::cpp::self_intersection_curves(bad_dynamic.mesh()),
                  std::out_of_range);
}
