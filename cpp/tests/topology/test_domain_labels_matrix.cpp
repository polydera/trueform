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
#include "carriers.hpp"
#include "mixed_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/topology/async/domain_labels.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Index, typename Real> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using mesh_type = tf::cpp::mesh<Index, Real, 3>;
  using result_type = tf::cpp::domain_labels_result<Index>;
};

using float_int32 = matrix_row<std::int32_t, float>;
using float_int64 = matrix_row<std::int64_t, float>;
using double_int32 = matrix_row<std::int32_t, double>;
using double_int64 = matrix_row<std::int64_t, double>;

template <typename Mesh, typename = void>
struct has_domain_labels : std::false_type {};

template <typename Mesh>
struct has_domain_labels<Mesh, std::void_t<decltype(tf::cpp::make_domain_labels(
                                   std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_domain_labels : std::false_type {};

template <typename Mesh>
struct has_async_domain_labels<
    Mesh, std::void_t<decltype(tf::cpp::async::make_domain_labels(
              std::declval<const Mesh &>()))>> : std::true_type {};

static_assert(has_domain_labels<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(has_domain_labels<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(!has_domain_labels<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(
    !has_domain_labels<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(
    has_async_domain_labels<tf::cpp::mesh<std::int64_t, float, 3>>::value);
static_assert(
    !has_async_domain_labels<tf::cpp::mesh<std::int32_t, double, 2>>::value);
static_assert(std::is_same_v<tf::cpp::domain_labels_result<std::int32_t>,
                             tf::cpp::domain_labels_result<>>);

template <typename Row> auto fixed_box() -> typename Row::owned_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1})};
}

/// The same box, stated with offsets that are i * 3 by construction.
template <typename Row>
auto dynamic_box() -> tf::cpp::test::mixed_mesh_of<typename Row::owned_type> {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  const auto fixed = fixed_box<Row>();
  const auto &faces = fixed.polygons.faces_buffer();
  std::vector<Index> offsets(faces.size() + 1);
  for (std::size_t face = 0; face != offsets.size(); ++face)
    offsets[face] = static_cast<Index>(face * 3);
  return {tf::cpp::test::polygons_of<Index, Real, 3>(
      offsets, faces.data_buffer(),
      fixed.polygons.points_buffer().data_buffer())};
}

template <typename Row, std::size_t Ngon>
auto empty_mesh() -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  return {};
}

template <typename Result> auto check_default_box(const Result &result) {
  using Index = std::decay_t<decltype(result.number_of_domains())>;
  CHECK(result.number_of_domains() == Index{2});
  CHECK(result.has_outer_shell_domain());
  CHECK(result.outer_shell_label() >= result.valid_label_begin());
  CHECK(result.outer_shell_label() < result.valid_label_end());
  CHECK(result.labels().raw_shape() == tf::small_vector<int, 3>{12, 2});

  bool carries_outer_shell = false;
  for (const auto label : result.labels()) {
    CHECK(label >= Index{0});
    CHECK(label < result.number_of_domains());
    CHECK(label != result.sentinel_label());
    carries_outer_shell |= label == result.outer_shell_label();
  }
  CHECK(carries_outer_shell);
}

} // namespace

TEMPLATE_TEST_CASE(
    "domain labels match the fixed Python dtype matrix and sentinel semantics",
    "[cpp][topology][domains][matrix][python-parity][fixed]", float_int32,
    float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Result = typename TestType::result_type;
  using function_type =
      Result (*)(const typename TestType::mesh_type &, tf::domain_config);
  const function_type symbol = &tf::cpp::make_domain_labels<Index, Real, 3>;
  REQUIRE(symbol != nullptr);

  auto box = fixed_box<TestType>();
  auto included = symbol(box.mesh(), tf::domain_config::none);
  static_assert(std::is_same_v<decltype(included), Result>);
  static_assert(
      std::is_same_v<decltype(included.labels()), tf::cpp::nd_array<Index>>);
  check_default_box(included);

  auto excluded = symbol(box.mesh(), tf::domain_config::exclude_outer_shell);
  CHECK(excluded.number_of_domains() == Index{1});
  CHECK(excluded.outer_shell_label() == excluded.number_of_domains());
  CHECK_FALSE(excluded.has_outer_shell_domain());
  bool carries_sentinel = false;
  for (const auto label : excluded.labels()) {
    CHECK(label >= Index{0});
    CHECK(label <= excluded.sentinel_label());
    carries_sentinel |= label == excluded.sentinel_label();
  }
  CHECK(carries_sentinel);
}

TEMPLATE_TEST_CASE("domain labels match the dynamic Python carrier matrix",
                   "[cpp][topology][domains][matrix][python-parity][dynamic]",
                   float_int32, float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  auto box = dynamic_box<TestType>();

  // AN ENTRY READS THE CACHE'S STRUCTURES: this one stands on the face
  // membership AND the edge link, and an untagged form would make core build
  // both and throw them away. The build counts are what state the difference.
  CHECK(box.cache.face_membership_build_count() == 0);
  CHECK(box.cache.manifold_edge_link_build_count() == 0);

  auto included = tf::cpp::make_domain_labels<Index, Real, 3>(box.mesh());
  check_default_box(included);
  CHECK(included.number_of_faces() ==
        static_cast<Index>(box.mesh().number_of_faces()));
  CHECK(box.cache.face_membership_build_count() == 1);
  CHECK(box.cache.manifold_edge_link_build_count() == 1);

  auto inferred = tf::cpp::make_domain_labels(box.mesh());
  CHECK(box.cache.face_membership_build_count() == 1);
  CHECK(box.cache.manifold_edge_link_build_count() == 1);
  static_assert(
      std::is_same_v<decltype(inferred), typename TestType::result_type>);
  check_default_box(inferred);

  auto excluded = tf::cpp::make_domain_labels<Index, Real, 3>(
      box.mesh(), tf::domain_config::exclude_outer_shell);
  CHECK(excluded.number_of_domains() == Index{1});
  CHECK(excluded.outer_shell_label() == excluded.sentinel_label());
}

TEMPLATE_TEST_CASE("domain labels return canonical typed empty results",
                   "[cpp][topology][domains][matrix][empty]", float_int32,
                   float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto states_nothing = [](const auto &owned) {
    auto result = tf::cpp::make_domain_labels(owned.mesh());
    CHECK(result.empty());
    CHECK(result.number_of_faces() == Index{0});
    CHECK(result.number_of_domains() == Index{0});
    CHECK(result.sentinel_label() == Index{0});
    CHECK(result.outer_shell_label() == Index{-1});
    CHECK_FALSE(result.has_outer_shell_domain());
    CHECK(result.labels().raw_shape() == tf::small_vector<int, 3>{0, 2});
  };
  states_nothing(empty_mesh<TestType, 3>());
  states_nothing(empty_mesh<TestType, tf::dynamic_size>());
  static_cast<void>(sizeof(Real));
}

TEMPLATE_TEST_CASE(
    "domain labels validate all generalized carriers and typed results",
    "[cpp][topology][domains][matrix][validation]", float_int32, float_int64,
    double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  // a caller holding nothing holds the EMPTY mesh, which states no domain
  typename TestType::owned_type nothing;
  CHECK(tf::cpp::make_domain_labels(nothing.mesh()).empty());

  auto box = fixed_box<TestType>();
  CHECK_THROWS_AS((tf::cpp::make_domain_labels<Index, Real, 3>(
                      box.mesh(), static_cast<tf::domain_config>(4))),
                  std::invalid_argument);

  // the library cannot see a caller's storage, so a face index is refused
  // where the READING is read
  auto bad = fixed_box<TestType>();
  tf::cpp::test::fill_storage(
      bad.polygons.points_buffer().data_buffer(),
      std::vector<Real>{Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}});
  bad.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::make_domain_labels(bad.mesh()), std::out_of_range);
}

TEST_CASE("int64 domain label results validate shape labels and access",
          "[cpp][topology][domains][typed-result][validation]") {
  using Result = tf::cpp::domain_labels_result<std::int64_t>;
  CHECK_THROWS_AS(Result(array<std::int64_t>({0, 1}, {2}), 2, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(Result(array<std::int64_t>({0, 3}, {1, 2}), 2, 0),
                  std::out_of_range);
  CHECK_THROWS_AS(Result(array<std::int64_t>({0, 1}, {1, 2}), -1, -1),
                  std::invalid_argument);

  Result result(array<std::int64_t>({0, 2}, {1, 2}), 2, 1);
  CHECK(result.label(0, 0) == 0);
  CHECK(result.label(0, 1) == 2);
  CHECK_THROWS_AS(result.label(-1, 0), std::out_of_range);
  CHECK_THROWS_AS(result.label(0, 2), std::out_of_range);
}

TEMPLATE_TEST_CASE(
    "async domain labels answer every fixed and dynamic Python carrier",
    "[cpp][topology][domains][matrix][async]", float_int32, float_int64,
    double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Result = typename TestType::result_type;
  const auto answers_from_the_mesh = [](const auto &owned) {
    auto pending = tf::cpp::async::make_domain_labels(owned.mesh());
    static_assert(std::is_same_v<decltype(pending), std::future<Result>>);
    check_default_box(pending.get());
  };
  answers_from_the_mesh(fixed_box<TestType>());
  answers_from_the_mesh(dynamic_box<TestType>());
  static_cast<void>(sizeof(Real));

  typename TestType::owned_type nothing;
  CHECK(tf::cpp::async::make_domain_labels(nothing.mesh()).get().empty());
}

TEST_CASE("domain labels submit exactly once at either arity",
          "[cpp][topology][domains][matrix][async][resolver]") {
  tf::cpp::test::counting_resolver resolver;

  auto dynamic = dynamic_box<double_int64>();
  check_default_box(tf::cpp::test::resolve_once(resolver, [&](auto counting) {
    return tf::cpp::async::make_domain_labels(counting, dynamic.mesh());
  }));

  auto fixed = fixed_box<double_int32>();
  check_default_box(tf::cpp::test::resolve_once(resolver, [&](auto counting) {
    return tf::cpp::async::make_domain_labels(counting, fixed.mesh());
  }));
  CHECK(resolver.submissions() == 2);
}

TEST_CASE("the domain label archive symbol answers the int32 axes",
          "[cpp][topology][domains][abi][archive-link]") {
  using Mesh = tf::cpp::mesh<tf::cpp::default_index_t, float>;
  using function_type =
      tf::cpp::domain_labels_result<> (*)(const Mesh &, tf::domain_config);
  const function_type generalized =
      &tf::cpp::make_domain_labels<std::int32_t, float, 3>;
  CHECK(generalized != nullptr);
}
