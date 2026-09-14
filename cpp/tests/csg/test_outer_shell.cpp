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
#include "trueform/cpp/csg/async/outer_shell.hpp"
#include "trueform/cpp/csg/outer_shell.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/reindex/concatenated.hpp"
#include "trueform/cpp/topology/is_closed.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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

template <typename Fn> struct mutating_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  Fn mutation;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutation();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Fn>
mutating_resolver(std::shared_ptr<std::atomic<int>>, Fn)
    -> mutating_resolver<Fn>;

template <typename Index, typename Real> struct outer_shell_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using result_type = tf::polygons_buffer<Index, Real, 3, 3>;
};

using outer_shell_float_int32 = outer_shell_row<std::int32_t, float>;
using outer_shell_float_int64 = outer_shell_row<std::int64_t, float>;
using outer_shell_double_int32 = outer_shell_row<std::int32_t, double>;
using outer_shell_double_int64 = outer_shell_row<std::int64_t, double>;

template <typename Index, typename Real, typename = void>
struct has_typed_outer_shell : std::false_type {};

template <typename Index, typename Real>
struct has_typed_outer_shell<
    Index, Real,
    std::void_t<decltype(tf::cpp::outer_shell<Index, Real>(
        std::declval<tf::cpp::mesh<Index, Real, 3, 3> &>()))>>
    : std::true_type {};

// The async entry takes the carrier the operation is about, so it is named by
// that carrier rather than by an index and a real of its own.
template <typename Index, typename Real, typename = void>
struct has_typed_async_outer_shell : std::false_type {};

template <typename Index, typename Real>
struct has_typed_async_outer_shell<
    Index, Real,
    std::void_t<decltype(tf::cpp::async::outer_shell(
        std::declval<tf::cpp::mesh<Index, Real, 3, 3> &>()))>>
    : std::true_type {};

template <typename Index, typename Real, typename = void>
struct has_resolver_typed_async_outer_shell : std::false_type {};

template <typename Index, typename Real>
struct has_resolver_typed_async_outer_shell<
    Index, Real,
    std::void_t<decltype(tf::cpp::async::outer_shell(
        std::declval<counting_resolver>(),
        std::declval<tf::cpp::mesh<Index, Real, 3, 3> &>()))>>
    : std::true_type {};

static_assert(has_typed_outer_shell<std::int32_t, float>::value);
static_assert(has_typed_outer_shell<std::int64_t, double>::value);
static_assert(has_typed_async_outer_shell<std::int64_t, float>::value);
static_assert(
    has_resolver_typed_async_outer_shell<std::int32_t, double>::value);

// An index that is not an index is refused by the carrier itself, so a mesh of
// one cannot be named to ask an entry about. What is left for these entries to
// refuse is a real they have no kernel for.
static_assert(!has_typed_outer_shell<std::int32_t, long double>::value);
static_assert(!has_typed_outer_shell<std::int64_t, int>::value);
static_assert(!has_typed_async_outer_shell<std::int32_t, long double>::value);
static_assert(!has_typed_async_outer_shell<std::int64_t, int>::value);
static_assert(
    !has_resolver_typed_async_outer_shell<std::int32_t, long double>::value);
static_assert(!has_resolver_typed_async_outer_shell<std::int64_t, int>::value);

/// The sixteen numbers a placement is, which is what an owner is placed by.
template <typename Real> auto translation(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Index, typename Real, std::size_t Ngon = 3>
using shell_owned = tf::cpp::test::owned_mesh<Index, Real, 3, Ngon>;

/// The same soup at either arity: a triangle mesh states the offsets it has by
/// construction.
template <typename Index, typename Real, std::size_t Ngon>
auto at_arity(tf::polygons_buffer<Index, Real, 3, 3> value)
    -> shell_owned<Index, Real, Ngon> {
  if constexpr (Ngon == 3) {
    return tf::cpp::test::operand_of(std::move(value));
  } else {
    tf::buffer<Index> offsets;
    offsets.allocate(value.size() + 1);
    for (std::size_t face = 0; face != offsets.size(); ++face)
      offsets[face] = static_cast<Index>(face * 3);
    return {tf::cpp::test::polygons_of<Index, Real>(
        offsets, value.faces_buffer().data_buffer(),
        value.points_buffer().data_buffer())};
  }
}

/// The same box, moved along x where it lies: a test writes through its own
/// storage and then states the change.
template <typename Index, typename Real>
auto shifted(tf::polygons_buffer<Index, Real, 3, 3> value, Real dx)
    -> shell_owned<Index, Real> {
  auto owner = tf::cpp::test::operand_of(std::move(value));
  for (auto point : owner.polygons.points())
    point[0] += dx;
  owner.cache.points_changed();
  return owner;
}

template <typename Real>
auto overlapping_boxes() -> shell_owned<tf::cpp::default_index_t, Real> {
  std::vector<shell_owned<tf::cpp::default_index_t, Real>> boxes;
  boxes.reserve(2);
  boxes.push_back(tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(Real{2}, Real{2}, Real{2})));
  boxes.push_back(
      shifted(tf::cpp::make_box_mesh(Real{2}, Real{2}, Real{2}), Real{1}));
  return tf::cpp::test::operand_of(tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<tf::cpp::default_index_t, Real, 3, 3>>{
          boxes[0].mesh(), boxes[1].mesh()}));
}

template <typename Real>
auto empty_mesh() -> shell_owned<tf::cpp::default_index_t, Real> {
  return {};
}

template <typename Row>
auto mixed_ngon_pyramid()
    -> tf::cpp::test::mixed_mesh_of<typename Row::mesh_type> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 4, 7, 10, 13, 16}, {0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}, Real{0.5}, Real{0.5}, Real{1}})};
}

template <typename Row, std::size_t Ngon = 3>
auto python_sphere_soup()
    -> shell_owned<typename Row::index_type, typename Row::real_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto first = tf::cpp::test::operand_of(
      tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 24, 24));
  const auto second =
      shifted(tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 24, 24), Real{1});
  return at_arity<Index, Real, Ngon>(
      tf::cpp::concatenate_meshes(std::vector<tf::cpp::mesh<Index, Real, 3, 3>>{
          first.mesh(), second.mesh()}));
}

template <typename Row, std::size_t Ngon = 3>
auto typed_overlapping_boxes()
    -> shell_owned<typename Row::index_type, typename Row::real_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto first = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh<Index, Real>(Real{2}, Real{2}, Real{2}));
  const auto second = shifted(
      tf::cpp::make_box_mesh<Index, Real>(Real{2}, Real{2}, Real{2}), Real{1});
  return at_arity<Index, Real, Ngon>(
      tf::cpp::concatenate_meshes(std::vector<tf::cpp::mesh<Index, Real, 3, 3>>{
          first.mesh(), second.mesh()}));
}
} // namespace

TEMPLATE_TEST_CASE("outer shell matches the Python fixed sphere matrix",
                   "[cpp][csg][outer-shell][matrix][python-parity][fixed]",
                   outer_shell_float_int32, outer_shell_float_int64,
                   outer_shell_double_int32, outer_shell_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Result = typename TestType::result_type;
  const auto sphere_volume = Real{4} / Real{3} * std::acos(Real{-1});

  const auto source = python_sphere_soup<TestType>();
  const auto source_faces = source.polygons.size();
  auto shell_polygons = tf::cpp::outer_shell<Index, Real>(source.mesh());
  static_assert(std::is_same_v<decltype(shell_polygons), Result>);
  const auto shell = tf::cpp::test::operand_of(std::move(shell_polygons));

  CHECK((tf::cpp::is_closed<Index, Real, 3>(shell.mesh())));
  const auto volume = tf::cpp::signed_volume<Index, Real, 3>(shell.mesh());
  CHECK(volume > sphere_volume);
  CHECK(volume < Real{2} * sphere_volume);
  CHECK(shell.polygons.size() < source_faces);

  const auto again = tf::cpp::test::operand_of(
      tf::cpp::outer_shell<Index, Real>(source.mesh()));
  CHECK((tf::cpp::is_closed<Index, Real, 3>(again.mesh())));
  CHECK(tf::cpp::signed_volume<Index, Real, 3>(again.mesh()) ==
        Catch::Approx(volume).epsilon(1e-4));
}

TEMPLATE_TEST_CASE("outer shell matches the Python dynamic sphere matrix",
                   "[cpp][csg][outer-shell][matrix][python-parity][dynamic]",
                   outer_shell_float_int32, outer_shell_float_int64,
                   outer_shell_double_int32, outer_shell_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto sphere_volume = Real{4} / Real{3} * std::acos(Real{-1});

  // an uncut face is emitted verbatim, so the shell states the arity its
  // operand did
  const auto source = python_sphere_soup<TestType, tf::dynamic_size>();
  auto shell_polygons = tf::cpp::outer_shell<Index, Real>(source.mesh());
  static_assert(
      std::is_same_v<decltype(shell_polygons),
                     tf::polygons_buffer<Index, Real, 3, tf::dynamic_size>>);
  const auto shell = tf::cpp::test::operand_of(std::move(shell_polygons));

  CHECK((tf::cpp::is_closed<Index, Real, 3>(shell.mesh())));
  const auto volume = tf::cpp::signed_volume<Index, Real, 3>(shell.mesh());
  CHECK(volume > sphere_volume);
  CHECK(volume < Real{2} * sphere_volume);
}

TEST_CASE("typed outer shell cold async reads a mixed-ngon dynamic input",
          "[cpp][csg][outer-shell][typed][dynamic][mixed-ngon][async][cold]") {
  using Row = outer_shell_double_int64;
  using Result = tf::polygons_buffer<std::int64_t, double, 3, tf::dynamic_size>;
  const auto source = mixed_ngon_pyramid<Row>();
  const auto source_offsets = tf::cpp::test::face_offsets_of(source.polygons);
  REQUIRE(source_offsets.length() == 6);
  REQUIRE(source_offsets[1] - source_offsets[0] == 4);
  REQUIRE(source_offsets[2] - source_offsets[1] == 3);
  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending = tf::cpp::async::outer_shell(counting_resolver{submissions},
                                             source.mesh());
  static_assert(std::is_same_v<decltype(pending), std::future<Result>>);

  const auto result = tf::cpp::test::operand_of(pending.get());
  CHECK(tf::cpp::is_closed(result.mesh()));
  CHECK(tf::cpp::signed_volume(result.mesh()) ==
        Catch::Approx(1.0 / 3.0).epsilon(1e-4));
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
}

TEST_CASE("outer shell drops open fragments and enclosed cavities",
          "[cpp][csg][outer-shell][semantics][open][cavity][python-parity]") {
  using Real = float;
  using Index = std::int64_t;
  using Mesh = tf::cpp::mesh<Index, Real, 3, 3>;

  const auto box = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh<Index, Real>(Real{2}, Real{2}, Real{2}));
  const shell_owned<Index, Real> fragment{
      tf::cpp::test::polygons_of<Index, Real>(
          {0, 1, 2}, {Real{-0.4}, Real{-0.4}, Real{0}, Real{0.4}, Real{-0.4},
                      Real{0}, Real{0}, Real{0.4}, Real{0}})};
  const auto open_soup = tf::cpp::test::operand_of(tf::cpp::concatenate_meshes(
      std::vector<Mesh>{box.mesh(), fragment.mesh()}));
  const auto expected_box_volume = tf::cpp::signed_volume(box.mesh());
  const auto open_shell = tf::cpp::test::operand_of(
      tf::cpp::outer_shell<Index, Real>(open_soup.mesh()));
  CHECK(tf::cpp::is_closed(open_shell.mesh()));
  CHECK(tf::cpp::signed_volume(open_shell.mesh()) ==
        Catch::Approx(expected_box_volume).epsilon(1e-4));

  const auto outer = tf::cpp::test::operand_of(
      tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 24, 24));
  const auto inner = tf::cpp::test::operand_of(
      tf::cpp::make_sphere_mesh<Index, Real>(Real{0.4}, 24, 24));
  const auto outer_volume = tf::cpp::signed_volume(outer.mesh());
  const auto cavity_soup =
      tf::cpp::test::operand_of(tf::cpp::concatenate_meshes(
          std::vector<Mesh>{outer.mesh(), inner.mesh()}));
  const auto cavity_shell = tf::cpp::test::operand_of(
      tf::cpp::outer_shell<Index, Real>(cavity_soup.mesh()));
  CHECK(tf::cpp::is_closed(cavity_shell.mesh()));
  CHECK(tf::cpp::signed_volume(cavity_shell.mesh()) ==
        Catch::Approx(outer_volume).epsilon(1e-4));
}

TEMPLATE_TEST_CASE(
    "outer shell typed empty validation and raw frame semantics",
    "[cpp][csg][outer-shell][matrix][empty][validation][transform]",
    outer_shell_float_int32, outer_shell_float_int64, outer_shell_double_int32,
    outer_shell_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using MixedResult = tf::polygons_buffer<Index, Real, 3, tf::dynamic_size>;

  const shell_owned<Index, Real> empty;
  const auto fixed_result = tf::cpp::outer_shell<Index, Real>(empty.mesh());
  CHECK(fixed_result.size() == 0);
  CHECK(fixed_result.points_buffer().size() == 0);

  // the shell states the arity its operand did
  const shell_owned<Index, Real, tf::dynamic_size> dynamic_empty;
  const auto dynamic_result =
      tf::cpp::outer_shell<Index, Real>(dynamic_empty.mesh());
  STATIC_REQUIRE(std::is_same_v<decltype(dynamic_result), const MixedResult>);
  CHECK(dynamic_result.size() == 0);

  // a face naming a point the geometry does not have is refused at the one
  // door the reading passes
  const shell_owned<Index, Real> malformed{
      tf::cpp::test::polygons_of<Index, Real>(
          {0, 1, 2}, {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}})};
  CHECK_THROWS_AS(tf::cpp::outer_shell(malformed.mesh()), std::out_of_range);

  // the shell is the operand's own boundary, so it is read where the operand
  // was authored rather than where its placement puts it
  auto transformed = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh<Index, Real>(Real{2}, Real{2}, Real{2}));
  transformed.place(translation(Real{10}));
  const auto raw = tf::cpp::test::operand_of(
      tf::cpp::outer_shell<Index, Real>(transformed.mesh()));
  CHECK(std::abs(tf::cpp::signed_volume(raw.mesh())) ==
        Catch::Approx(Real{8}).epsilon(1e-4));
  for (const auto point : raw.polygons.points())
    for (const auto coordinate : point)
      CHECK(std::abs(coordinate) <= Real{1});

  auto bad_intersect = tf::intersect_config{
      tf::intersect_mode::primitives | tf::intersect_mode::resolve_contours};
  bad_intersect.tolerance = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(
      (tf::cpp::outer_shell<Index, Real>(transformed.mesh(), bad_intersect)),
      std::invalid_argument);
}

TEST_CASE("typed outer shell async answers from the reading it holds",
          "[cpp][csg][outer-shell][typed][async][cache]") {
  using Row = outer_shell_double_int64;
  using Result = tf::polygons_buffer<std::int64_t, double, 3, tf::dynamic_size>;
  auto source = typed_overlapping_boxes<Row, tf::dynamic_size>();
  auto retained_placement = translation(0.0);
  source.place(retained_placement);

  const auto prewarmed =
      tf::cpp::outer_shell<std::int64_t, double>(source.mesh());
  CHECK(prewarmed.size() > 0);

  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending =
      tf::cpp::async::outer_shell(mutating_resolver{submissions,
                                                    [&] {
                                                      // the placement was
                                                      // copied into the
                                                      // fixture's own slot
                                                      retained_placement[3] =
                                                          20.0;
                                                    }},
                                  source.mesh());
  static_assert(std::is_same_v<decltype(pending), std::future<Result>>);

  const auto result = tf::cpp::test::operand_of(pending.get());
  CHECK(tf::cpp::is_closed(result.mesh()));
  CHECK(std::abs(tf::cpp::signed_volume(result.mesh())) ==
        Catch::Approx(12.0).epsilon(1e-4));
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
}

TEMPLATE_TEST_CASE("outer shell repairs an overlapping closed mesh",
                   "[cpp][csg][outer-shell][sync]", float, double) {
  const auto source = overlapping_boxes<TestType>();
  auto shell_polygons = tf::cpp::outer_shell(source.mesh());

  static_assert(std::is_same_v<
                decltype(shell_polygons),
                tf::polygons_buffer<tf::cpp::default_index_t, TestType, 3, 3>>);
  CHECK(shell_polygons.size() > 0);
  CHECK(shell_polygons.points_buffer().size() > 0);
  const auto shell = tf::cpp::test::operand_of(std::move(shell_polygons));
  CHECK(tf::cpp::is_closed(shell.mesh()));
  CHECK(std::abs(tf::cpp::signed_volume(shell.mesh())) ==
        Catch::Approx(TestType{12}).epsilon(1e-4));
}

TEMPLATE_TEST_CASE("outer shell ignores transforms",
                   "[cpp][csg][outer-shell][transform]", float, double) {
  auto source = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  source.place(translation(TestType{10}));

  const auto shell =
      tf::cpp::test::operand_of(tf::cpp::outer_shell(source.mesh()));

  CHECK(std::abs(tf::cpp::signed_volume(shell.mesh())) ==
        Catch::Approx(TestType{8}).epsilon(1e-4));
  for (const auto point : shell.polygons.points())
    for (const auto coordinate : point)
      CHECK(std::abs(coordinate) <= TestType{1});
}

TEMPLATE_TEST_CASE("outer shell defines empty and invalid input behavior",
                   "[cpp][csg][outer-shell][validation]", float, double) {
  const auto empty = empty_mesh<TestType>();
  const auto empty_shell = tf::cpp::outer_shell(empty.mesh());
  CHECK(empty_shell.size() == 0);
  CHECK(empty_shell.points_buffer().size() == 0);

  // a default-constructed owner is the EMPTY mesh, whose shell is empty too
  const shell_owned<tf::cpp::default_index_t, TestType> nothing;
  CHECK(tf::cpp::outer_shell(nothing.mesh()).size() == 0);

  auto bad_indices = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  bad_indices.polygons.points_buffer().data_buffer().clear();
  bad_indices.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::outer_shell(bad_indices.mesh()), std::out_of_range);

  const auto valid = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  auto bad_intersect = tf::intersect_config{
      tf::intersect_mode::primitives | tf::intersect_mode::resolve_contours};
  bad_intersect.tolerance = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(tf::cpp::outer_shell(valid.mesh(), bad_intersect),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("outer shell async borrows its input and submits once",
                   "[cpp][csg][outer-shell][async]", float, double) {
  using Result = tf::polygons_buffer<tf::cpp::default_index_t, TestType, 3, 3>;
  const auto source = overlapping_boxes<TestType>();
  auto pending = tf::cpp::async::outer_shell(source.mesh());
  static_assert(std::is_same_v<decltype(pending), std::future<Result>>);

  const auto result = tf::cpp::test::operand_of(pending.get());
  CHECK(tf::cpp::is_closed(result.mesh()));
  CHECK(std::abs(tf::cpp::signed_volume(result.mesh())) ==
        Catch::Approx(TestType{12}).epsilon(1e-4));

  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto custom_source = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  auto custom = tf::cpp::async::outer_shell(counting_resolver{submissions},
                                            custom_source.mesh());
  static_assert(std::is_same_v<decltype(custom), std::future<Result>>);
  CHECK(custom.get().size() > 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);

  const shell_owned<tf::cpp::default_index_t, TestType> nothing;
  CHECK(tf::cpp::async::outer_shell(nothing.mesh()).get().size() == 0);
}
