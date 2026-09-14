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
#include "trueform/cpp/topology.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

template <typename Index, typename Real, std::size_t Dims> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr std::size_t dims = Dims;
};

using matrix_rows = std::tuple<
    matrix_row<std::int32_t, float, 2>, matrix_row<std::int32_t, float, 3>,
    matrix_row<std::int64_t, float, 2>, matrix_row<std::int64_t, float, 3>,
    matrix_row<std::int32_t, double, 2>, matrix_row<std::int32_t, double, 3>,
    matrix_row<std::int64_t, double, 2>, matrix_row<std::int64_t, double, 3>>;

template <typename T>
auto array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> data;
  data.allocate(values.size());
  std::copy(values.begin(), values.end(), data.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(data), std::move(shape));
}

template <typename Index>
auto blocks(std::initializer_list<Index> offsets,
            std::initializer_list<Index> data)
    -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      array<Index>(offsets, {static_cast<int>(offsets.size())}),
      array<Index>(data, {static_cast<int>(data.size())}));
}

template <typename Index>
auto fixture_connectivity() -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return blocks<Index>({0, 2, 5, 8, 10}, {1, 2, 0, 2, 3, 0, 1, 3, 1, 2});
}

template <typename Row>
auto fixture_points() -> tf::cpp::nd_array<typename Row::real_type> {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return array<Real>({0, 0, 1, 0, Real{0.5}, 1, Real{1.5}, 1}, {4, 2});
  else
    return array<Real>({0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{1.5}, 1, 0},
                       {4, 3});
}

template <typename Index>
auto equals(const tf::cpp::offset_blocked_buffer<Index, Index> &actual,
            std::initializer_list<int> offsets, std::initializer_list<int> data)
    -> bool {
  const auto equal_values = [](const tf::cpp::nd_array<Index> &values,
                               std::initializer_list<int> expected) {
    return values.length() == expected.size() &&
           std::equal(values.begin(), values.end(), expected.begin(),
                      [](Index left, int right) {
                        return left == static_cast<Index>(right);
                      });
  };
  return equal_values(actual.offsets(), offsets) &&
         equal_values(actual.data(), data);
}

template <typename Index>
auto is_subset(const tf::cpp::nd_array<Index> &subset,
               const tf::cpp::nd_array<Index> &superset) -> bool {
  return std::all_of(subset.begin(), subset.end(), [&](Index value) {
    return std::find(superset.begin(), superset.end(), value) != superset.end();
  });
}

struct mutating_resolver {
  int *submissions;
  std::function<void()> mutate;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*submissions;
    mutate();
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "raw k-rings and metric neighborhoods match exact Python fixtures",
    "[cpp][topology][neighborhoods][python-parity][matrix]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto connectivity = fixture_connectivity<Index>();
  auto points = fixture_points<TestType>();

  const auto k1 = tf::cpp::k_rings(connectivity, 1);
  STATIC_REQUIRE(
      std::is_same_v<decltype(k1),
                     const tf::cpp::offset_blocked_buffer<Index, Index>>);
  CHECK(equals(k1, {0, 2, 5, 8, 10}, {1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));

  const auto k2 = tf::cpp::k_rings(connectivity, 2);
  CHECK(equals(k2, {0, 3, 6, 9, 12}, {1, 2, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0}));
  for (int seed = 0; seed < connectivity.size(); ++seed)
    CHECK(is_subset(k1.get(seed), k2.get(seed)));

  const auto inclusive = tf::cpp::k_rings(connectivity, 1, true);
  CHECK(equals(inclusive, {0, 3, 7, 11, 14},
               {0, 1, 2, 1, 0, 2, 3, 2, 0, 1, 3, 3, 1, 2}));

  const auto small = tf::cpp::neighborhoods<Index, Real, Dims>(
      connectivity, points, Real{0.5});
  CHECK(equals(small, {0, 0, 0, 0, 0}, {}));

  const auto metric = tf::cpp::neighborhoods<Index, Real, Dims>(
      connectivity, points, Real{1.5});
  CHECK(equals(metric, {0, 2, 5, 8, 10}, {1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));
  const auto metric_large =
      tf::cpp::neighborhoods<Index, Real, Dims>(connectivity, points, Real{2});
  CHECK(equals(metric_large, {0, 3, 6, 9, 12},
               {1, 2, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0}));
  for (int seed = 0; seed < connectivity.size(); ++seed)
    CHECK(is_subset(metric.get(seed), metric_large.get(seed)));

  const auto metric_inclusive = tf::cpp::neighborhoods<Index, Real, Dims>(
      connectivity, points, Real{1.5}, true);
  CHECK(equals(metric_inclusive, {0, 3, 7, 11, 14},
               {0, 1, 2, 1, 0, 2, 3, 2, 0, 1, 3, 3, 1, 2}));
}

TEMPLATE_LIST_TEST_CASE("raw neighborhoods return canonical empty owners",
                        "[cpp][topology][neighborhoods][empty][ownership]",
                        matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto empty = blocks<Index>({0}, {});
  auto noncanonical_empty = blocks<Index>({}, {});
  auto points = array<Real>({}, {0, static_cast<int>(Dims)});

  const auto rings = tf::cpp::k_rings(empty, 1);
  const auto noncanonical_rings = tf::cpp::k_rings(noncanonical_empty, 1);
  const auto metric =
      tf::cpp::neighborhoods<Index, Real, Dims>(empty, points, Real{1});
  const auto noncanonical_metric = tf::cpp::neighborhoods<Index, Real, Dims>(
      noncanonical_empty, points, Real{1});
  // an empty connectivity has no neighbourhoods however it spelled itself
  CHECK(equals(rings, {}, {}));
  CHECK(equals(noncanonical_rings, {}, {}));
  CHECK(equals(metric, {}, {}));
  CHECK(equals(noncanonical_metric, {}, {}));
  CHECK(rings.offsets().raw_owner() != empty.offsets().raw_owner());
  CHECK(metric.offsets().raw_owner() != empty.offsets().raw_owner());
}

TEMPLATE_LIST_TEST_CASE(
    "raw neighborhoods validate connectivity points and parameters",
    "[cpp][topology][neighborhoods][validation]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto connectivity = fixture_connectivity<Index>();
  auto points = fixture_points<TestType>();
  tf::cpp::offset_blocked_buffer<Index, Index> invalid_connectivity;
  CHECK_THROWS_AS(tf::cpp::k_rings(invalid_connectivity, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::k_rings(connectivity, 0), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::k_rings(connectivity, -1), std::invalid_argument);

  auto negative_peer = blocks<Index>({0, 1}, {Index{-1}});
  auto high_peer = blocks<Index>({0, 1}, {Index{1}});
  CHECK_THROWS_AS(tf::cpp::k_rings(negative_peer, 1), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::k_rings(high_peer, 1), std::out_of_range);

  auto nonzero_offset = fixture_connectivity<Index>();
  nonzero_offset.offsets()[0] = Index{1};
  CHECK_THROWS_AS(tf::cpp::k_rings(nonzero_offset, 1), std::invalid_argument);
  // an entry refuses in the job it was dispatched as, and the future carries
  // the refusal
  CHECK_THROWS_AS(tf::cpp::async::k_rings(nonzero_offset, 1).get(),
                  std::invalid_argument);
  auto decreasing_offsets = fixture_connectivity<Index>();
  decreasing_offsets.offsets()[2] = Index{0};
  CHECK_THROWS_AS(tf::cpp::k_rings(decreasing_offsets, 1),
                  std::invalid_argument);
  auto excessive_end = fixture_connectivity<Index>();
  excessive_end.offsets()[excessive_end.offsets().length() - 1] += Index{1};
  CHECK_THROWS_AS(tf::cpp::k_rings(excessive_end, 1), std::invalid_argument);

  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      invalid_connectivity, points, Real{1})),
                  std::invalid_argument);
  tf::cpp::nd_array<Real> invalid_points;
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      connectivity, invalid_points, Real{1})),
                  std::invalid_argument);
  auto flat_points = array<Real>({0, 0, 1, 0}, {4});
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      connectivity, flat_points, Real{1})),
                  std::invalid_argument);
  auto wrong_dims =
      array<Real>({}, {0, static_cast<int>(Dims == std::size_t{2} ? 3 : 2)});
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      connectivity, wrong_dims, Real{1})),
                  std::invalid_argument);
  auto wrong_count = array<Real>({}, {0, static_cast<int>(Dims)});
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      connectivity, wrong_count, Real{1})),
                  std::invalid_argument);
  auto one_point =
      array<Real>(Dims == std::size_t{2} ? std::initializer_list<Real>{0, 0}
                                         : std::initializer_list<Real>{0, 0, 0},
                  {1, static_cast<int>(Dims)});
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      negative_peer, one_point, Real{1})),
                  std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(
                      high_peer, one_point, Real{1})),
                  std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(connectivity,
                                                             points, Real{0})),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::neighborhoods<Index, Real, Dims>(connectivity,
                                                             points, Real{-1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::neighborhoods<Index, Real, Dims>(
          connectivity, points, std::numeric_limits<Real>::quiet_NaN())),
      std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "raw neighborhood sync and async results own typed storage",
    "[cpp][topology][neighborhoods][async][deep-ownership]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using result_type = tf::cpp::offset_blocked_buffer<Index, Index>;

  auto sync_connectivity = fixture_connectivity<Index>();
  auto sync_points = fixture_points<TestType>();
  auto sync_result = tf::cpp::neighborhoods<Index, Real, Dims>(
      sync_connectivity, sync_points, Real{1.5});
  CHECK(sync_result.offsets().raw_owner() !=
        sync_connectivity.offsets().raw_owner());
  CHECK(sync_result.data().raw_owner() != sync_connectivity.data().raw_owner());
  sync_connectivity.destroy();
  sync_points.destroy();
  CHECK(equals(sync_result, {0, 2, 5, 8, 10}, {1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));

  int ring_submissions = 0;
  auto ring_connectivity = fixture_connectivity<Index>();
  auto ring_future = tf::cpp::async::k_rings(
      mutating_resolver{&ring_submissions,
                        [&ring_connectivity] { ring_connectivity.destroy(); }},
      ring_connectivity, 1);
  STATIC_REQUIRE(
      std::is_same_v<decltype(ring_future), std::future<result_type>>);
  CHECK(ring_submissions == 1);
  CHECK_FALSE(ring_connectivity.is_valid());
  CHECK(equals(ring_future.get(), {0, 2, 5, 8, 10},
               {1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));

  int metric_submissions = 0;
  auto metric_connectivity = fixture_connectivity<Index>();
  auto metric_points = fixture_points<TestType>();
  auto metric_future = tf::cpp::async::neighborhoods<Index, Real, Dims>(
      mutating_resolver{&metric_submissions,
                        [&] {
                          metric_connectivity.destroy();
                          metric_points.destroy();
                        }},
      metric_connectivity, metric_points, Real{1.5});
  STATIC_REQUIRE(
      std::is_same_v<decltype(metric_future), std::future<result_type>>);
  CHECK(metric_submissions == 1);
  CHECK_FALSE(metric_connectivity.is_valid());
  CHECK_FALSE(metric_points.is_valid());
  CHECK(equals(metric_future.get(), {0, 2, 5, 8, 10},
               {1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));
}
