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
#include "trueform/cpp/csg.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/geometry/volume.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
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

template <typename Index, typename Real>
using csg_owned = tf::cpp::test::owned_mesh<Index, Real, 3>;

/// The offset box the fixtures overlap with: a test writes through its own
/// storage and then states the change, which is the protocol itself.
template <typename Index, typename Real>
auto offset_box() -> csg_owned<Index, Real> {
  csg_owned<Index, Real> box{
      tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1})};
  for (auto point : box.polygons.points()) {
    point[0] += Real{0.4};
    point[1] += Real{0.3};
    point[2] += Real{0.2};
  }
  box.cache.points_changed();
  return box;
}

template <typename Real>
auto boxes() -> std::vector<csg_owned<tf::cpp::default_index_t, Real>> {
  std::vector<csg_owned<tf::cpp::default_index_t, Real>> result;
  result.reserve(2);
  result.push_back({tf::cpp::make_box_mesh(Real{1}, Real{1}, Real{1})});
  result.push_back(offset_box<tf::cpp::default_index_t, Real>());
  return result;
}

/// The sheet and the box the csg documentation states its selection reads
/// against: a 2x2 separator at z = 0 through the unit box it cuts, so the
/// cap is area one, the annulus three, and each read has an exact answer.
template <typename Real>
auto sheet_and_box() -> std::vector<csg_owned<tf::cpp::default_index_t, Real>> {
  std::vector<csg_owned<tf::cpp::default_index_t, Real>> result;
  result.reserve(2);
  result.push_back({tf::cpp::make_plane_mesh(Real{2}, Real{2})});
  result.push_back({tf::cpp::make_box_mesh(Real{1}, Real{1}, Real{1})});
  return result;
}

/// The views an operand list is, over owners the caller keeps: the vector is
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

/// A result is core's own storage, so a check reads its arrays where they lie.
template <typename Index, typename Real, std::size_t Ngon>
auto result_points(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> tf::cpp::nd_array<Real> {
  const auto &storage = value.points_buffer().data_buffer();
  const auto count = static_cast<int>(value.points_buffer().size());
  return tf::cpp::nd_array<Real>::from_borrowed(
      {}, const_cast<Real *>(storage.data()), storage.size(), {count, 3});
}

template <typename Index, typename Real, std::size_t Ngon>
auto volume(const tf::polygons_buffer<Index, Real, 3, Ngon> &value) -> Real {
  const csg_owned<Index, Real> owner{value};
  return std::abs(tf::cpp::signed_volume(owner.mesh()));
}

template <typename Index, typename Real, std::size_t Ngon>
auto surface_area(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> Real {
  tf::cpp::cache<Index, Real, 3, Ngon> cache;
  return tf::cpp::area(tf::cpp::test::reading_over(value, cache));
}

/// The two sheet reads carry the same pieces with opposite windings, so the
/// one component a surface at z = 0 has says which: twice its signed area.
template <typename Index, typename Real>
auto sheet_signed_area(const tf::polygons_buffer<Index, Real, 3, 3> &value)
    -> Real {
  auto total = Real{0};
  for (const auto face : value.polygons()) {
    const auto first = face[1] - face[0];
    const auto second = face[2] - face[0];
    total += first[0] * second[1] - first[1] * second[0];
  }
  return total / Real{2};
}

template <typename Real>
auto check_domain_index_map_alignment(
    const tf::cpp::csg_domains_index_map_result<tf::cpp::default_index_t, Real>
        &result,
    std::size_t number_of_cells, int number_of_tags) -> void {
  CHECK(result.meshes.size() == number_of_cells);
  CHECK(result.ids.length() == number_of_cells);
  CHECK(result.number_of_tags == number_of_tags);
  CHECK(result.inclusion.raw_shape() ==
        tf::small_vector<int, 3>{static_cast<int>(number_of_cells),
                                 number_of_tags});
  CHECK(result.inclusion.length() ==
        number_of_cells * static_cast<std::size_t>(number_of_tags));
  for (const auto value : result.inclusion)
    CHECK((value == std::int8_t{0} || value == std::int8_t{1}));

  const auto check_offsets = [number_of_cells](const auto &offsets,
                                               const auto &data) {
    if (number_of_cells == 0) {
      CHECK(data.empty());
      REQUIRE(offsets.length() <= 1);
      if (!offsets.empty())
        CHECK(offsets[0] == 0);
      return;
    }
    REQUIRE(offsets.length() == number_of_cells + 1);
    CHECK(offsets[0] == 0);
    CHECK(offsets[number_of_cells] == static_cast<std::int32_t>(data.length()));
  };
  check_offsets(result.face_tag_offsets, result.face_tag_data);
  check_offsets(result.face_offsets, result.face_data);
  check_offsets(result.point_tag_offsets, result.point_tag_data);
  check_offsets(result.point_offsets, result.point_data);
}

template <typename Real>
auto inclusion_column_count(
    const tf::cpp::csg_domains_index_map_result<tf::cpp::default_index_t, Real>
        &result,
    int column) -> std::size_t {
  auto count = std::size_t{0};
  for (auto row = 0; row < result.inclusion.shape_at(0); ++row)
    count +=
        result.inclusion[static_cast<std::size_t>(row) * result.number_of_tags +
                         static_cast<std::size_t>(column)] == std::int8_t{1};
  return count;
}

template <typename Index, typename Real> struct csg_matrix_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = csg_owned<Index, Real>;
  using view_type = tf::cpp::mesh<Index, Real, 3, 3>;
  using graph_type = tf::cpp::csg_graph<Index, Real>;
};

using float_int32 = csg_matrix_row<std::int32_t, float>;
using float_int64 = csg_matrix_row<std::int64_t, float>;
using double_int32 = csg_matrix_row<std::int32_t, double>;
using double_int64 = csg_matrix_row<std::int64_t, double>;

template <typename Row>
auto matrix_boxes() -> std::vector<typename Row::mesh_type> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  std::vector<typename Row::mesh_type> result;
  result.reserve(2);
  result.push_back(
      {tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1})});
  result.push_back(offset_box<Index, Real>());
  return result;
}

template <typename T>
auto arrays_equal(const tf::cpp::nd_array<T> &left,
                  const tf::cpp::nd_array<T> &right) -> bool {
  if (left.raw_shape() != right.raw_shape() || left.length() != right.length())
    return false;
  for (std::size_t i = 0; i < left.length(); ++i)
    if (left[i] != right[i])
      return false;
  return true;
}

template <typename Operands, typename = void>
struct csg_graph_takes : std::false_type {};

template <typename Operands>
struct csg_graph_takes<Operands, std::void_t<decltype(tf::cpp::make_csg_graph(
                                     std::declval<const Operands &>()))>>
    : std::true_type {};

using int32_float_graph = tf::cpp::csg_graph<std::int32_t, float>;
using int64_double_graph = tf::cpp::csg_graph<std::int64_t, double>;

static_assert(
    std::is_same_v<tf::curves_buffer<std::int32_t, float, 3>,
                   tf::curves_buffer<tf::cpp::default_index_t, float, 3>>);
static_assert(std::is_same_v<tf::curves_buffer<std::int64_t, double, 3>,
                             tf::curves_buffer<std::int64_t, double, 3>>);
static_assert(
    std::is_same_v<decltype(tf::cpp::make_csg_graph(
                       std::declval<std::vector<
                           tf::cpp::mesh<std::int32_t, float, 3, 3>>>())),
                   int32_float_graph>);
static_assert(
    std::is_same_v<decltype(tf::cpp::make_csg_graph(
                       std::declval<std::vector<
                           tf::cpp::mesh<std::int64_t, double, 3, 3>>>())),
                   int64_double_graph>);
static_assert(
    std::is_same_v<decltype(tf::cpp::csg_intersection_curves(
                       std::declval<const int32_float_graph &>())),
                   tf::curves_buffer<tf::cpp::default_index_t, float, 3>>);
static_assert(std::is_same_v<decltype(tf::cpp::csg_intersection_curves(
                                 std::declval<const int64_double_graph &>())),
                             tf::curves_buffer<std::int64_t, double, 3>>);
static_assert(
    std::is_same_v<
        tf::cpp::csg_mesh_labeled_result<std::int32_t, float>,
        tf::cpp::csg_mesh_labeled_result<tf::cpp::default_index_t, float>>);
static_assert(std::is_same_v<
              tf::cpp::csg_domains_index_map_result<std::int64_t, double>,
              tf::cpp::csg_domains_index_map_result<std::int64_t, double>>);

} // namespace

TEMPLATE_TEST_CASE("csg graph builds once and serves every mesh result",
                   "[cpp][csg][graph][sync]", float, double) {
  const auto owners = boxes<TestType>();
  const auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
  const auto union_expression = tf::csg::op(0) | tf::csg::op(1);
  const auto difference_expression = tf::csg::op(0) - tf::csg::op(1);

  const auto full = tf::cpp::make_csg_mesh(graph);
  const auto merged = tf::cpp::make_csg_mesh(graph, union_expression);
  const auto difference = tf::cpp::make_csg_mesh(graph, difference_expression);
  const auto labeled =
      tf::cpp::make_csg_mesh_with_labels(graph, union_expression);
  const auto mapped =
      tf::cpp::make_csg_mesh_with_index_map(graph, union_expression);
  const auto points = tf::cpp::csg_created_points(graph);
  const auto curves = tf::cpp::csg_intersection_curves(graph);

  CHECK(graph.is_valid());
  CHECK(full.size() > merged.size());
  CHECK(volume(merged) > volume(difference));
  CHECK(labeled.tag_labels.length() == labeled.mesh.size());
  CHECK(labeled.face_labels.length() == labeled.tag_labels.length());
  CHECK(mapped.point_tag_labels.length() == mapped.mesh.points_buffer().size());
  CHECK(mapped.face_tag_labels.length() == mapped.mesh.size());
  CHECK(mapped.number_of_tags == 2);
  CHECK(mapped.point_f_offsets.length() == 3);
  CHECK(mapped.uncut_faces.raw_shape() == tf::small_vector<int, 3>{2, 2});
  for (std::size_t tag = 0; tag < 2; ++tag) {
    const auto begin = mapped.uncut_faces[tag * 2];
    const auto end = mapped.uncut_faces[tag * 2 + 1];
    CHECK(begin <= end);
    CHECK(static_cast<std::size_t>(end) <= mapped.mesh.size());
    for (auto face = begin; face < end; ++face)
      CHECK(static_cast<std::size_t>(
                mapped.face_tag_labels[static_cast<std::size_t>(face)]) == tag);
  }
  CHECK(points.raw_shape() == tf::small_vector<int, 3>{points.shape_at(0), 3});
  CHECK(points.shape_at(0) > 0);
  CHECK(curves.size() > 0);
}

TEMPLATE_TEST_CASE("csg graph domain result carriers stay aligned",
                   "[cpp][csg][graph][domains]", float, double) {
  const auto retained_overlap_inclusion = [] {
    const auto owners = boxes<TestType>();
    auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
    const auto overlap_expression = tf::csg::op(0) & tf::csg::op(1);
    const auto config = tf::domain_config::exclude_outer_shell |
                        tf::domain_config::ignore_open_fragments;

    const auto plain = tf::cpp::make_csg_domains(graph, config);
    const auto overlap =
        tf::cpp::make_csg_domains(graph, overlap_expression, config);
    const auto inside_first =
        tf::cpp::make_csg_domains(graph, tf::csg::op(0), config);
    const auto inside_second =
        tf::cpp::make_csg_domains(graph, tf::csg::op(1), config);
    const auto labeled = tf::cpp::make_csg_domains_with_labels(graph, config);
    const auto mapped = tf::cpp::make_csg_domains_with_index_map(graph, config);
    auto filtered = tf::cpp::make_csg_domains_with_index_map(
        graph, overlap_expression, config);
    const auto empty = tf::cpp::make_csg_domains_with_index_map(
        graph, tf::csg::op(0) & ~tf::csg::op(0), config);

    REQUIRE(plain.meshes.size() == 3);
    REQUIRE(overlap.meshes.size() == 1);
    CHECK(labeled.meshes.size() == plain.meshes.size());
    CHECK(labeled.ids.length() == plain.meshes.size());
    REQUIRE(labeled.tag_offsets.length() == plain.meshes.size() + 1);
    REQUIRE(labeled.face_offsets.length() == plain.meshes.size() + 1);
    CHECK(labeled.tag_offsets[0] == 0);
    CHECK(labeled.face_offsets[0] == 0);

    check_domain_index_map_alignment(mapped, plain.meshes.size(), 2);
    check_domain_index_map_alignment(filtered, overlap.meshes.size(), 2);
    check_domain_index_map_alignment(empty, 0, 2);
    CHECK(inclusion_column_count(mapped, 0) == inside_first.meshes.size());
    CHECK(inclusion_column_count(mapped, 1) == inside_second.meshes.size());
    auto mapped_overlap_count = std::size_t{0};
    for (auto row = 0; row < mapped.inclusion.shape_at(0); ++row) {
      const auto offset = static_cast<std::size_t>(row) * 2;
      mapped_overlap_count += mapped.inclusion[offset] == std::int8_t{1} &&
                              mapped.inclusion[offset + 1] == std::int8_t{1};
    }
    CHECK(mapped_overlap_count == overlap.meshes.size());
    CHECK(inclusion_column_count(filtered, 0) == overlap.meshes.size());
    CHECK(inclusion_column_count(filtered, 1) == overlap.meshes.size());
    REQUIRE(filtered.inclusion.length() == 2);
    CHECK(filtered.inclusion[0] == std::int8_t{1});
    CHECK(filtered.inclusion[1] == std::int8_t{1});

    auto retained = filtered.inclusion.shallow_copy();
    filtered.inclusion.destroy();
    graph.destroy();
    return retained;
  }();

  REQUIRE(retained_overlap_inclusion.raw_shape() ==
          tf::small_vector<int, 3>{1, 2});
  REQUIRE(retained_overlap_inclusion.length() == 2);
  CHECK(retained_overlap_inclusion[0] == std::int8_t{1});
  CHECK(retained_overlap_inclusion[1] == std::int8_t{1});
}

TEMPLATE_TEST_CASE("csg graph answers the sheet selection reads",
                   "[cpp][csg][graph][selection]", float, double) {
  const auto owners = sheet_and_box<TestType>();
  const auto graph = tf::cpp::make_csg_graph(meshes_of(owners), {0});
  const auto above_and_outside = ~tf::csg::op(0) & ~tf::csg::op(1);

  const auto annulus =
      tf::cpp::make_csg_mesh(graph, tf::csg::selection({0}, above_and_outside));
  const auto cap =
      tf::cpp::make_csg_mesh(graph, tf::csg::inside({0}, tf::csg::op(1)));
  const auto annulus_inside =
      tf::cpp::make_csg_mesh(graph, tf::csg::inside({0}, ~tf::csg::op(1)));
  const auto bounding_the_box =
      tf::cpp::make_csg_mesh(graph, tf::csg::selection({0}, tf::csg::op(1)));
  const auto walls_below =
      tf::cpp::make_csg_mesh(graph, tf::csg::inside({1}, tf::csg::op(0)));
  const auto sheet_inside_itself =
      tf::cpp::make_csg_mesh(graph, tf::csg::inside({0}, tf::csg::op(0)));

  CHECK(static_cast<double>(surface_area(annulus)) ==
        Catch::Approx(3.0).margin(1e-5));
  CHECK(static_cast<double>(surface_area(cap)) ==
        Catch::Approx(1.0).margin(1e-5));
  CHECK(static_cast<double>(surface_area(annulus_inside)) ==
        Catch::Approx(3.0).margin(1e-5));
  CHECK(static_cast<double>(surface_area(walls_below)) ==
        Catch::Approx(3.0).margin(1e-5));
  // both sides of each sheet piece answer the same, so neither bounds
  CHECK(bounding_the_box.size() == 0);
  CHECK(sheet_inside_itself.size() == 0);

  // a boundary read winds away from its region, an inside read keeps the
  // sheet's own +z
  CHECK(static_cast<double>(sheet_signed_area(annulus)) ==
        Catch::Approx(-3.0).margin(1e-5));
  CHECK(static_cast<double>(sheet_signed_area(cap)) ==
        Catch::Approx(1.0).margin(1e-5));
  CHECK(static_cast<double>(sheet_signed_area(annulus_inside)) ==
        Catch::Approx(3.0).margin(1e-5));

  const tf::csg::selection_t cap_read = tf::csg::inside({0}, tf::csg::op(1));
  const auto labeled = tf::cpp::make_csg_mesh_with_labels(graph, cap_read);
  const auto mapped = tf::cpp::make_csg_mesh_with_index_map(graph, cap_read);
  CHECK(labeled.mesh.size() == cap.size());
  REQUIRE(labeled.tag_labels.length() == cap.size());
  CHECK(labeled.face_labels.length() == labeled.tag_labels.length());
  for (const auto tag : labeled.tag_labels)
    CHECK(tag == tf::cpp::default_index_t{0});
  CHECK(mapped.mesh.size() == cap.size());
  CHECK(mapped.face_tag_labels.length() == cap.size());
  CHECK(mapped.point_tag_labels.length() == mapped.mesh.points_buffer().size());
  CHECK(mapped.number_of_tags == 2);

  // an expression IS a selection: the boundary read of every form
  const auto carved = tf::csg::op(1) - tf::csg::op(0);
  const auto from_expression = tf::cpp::make_csg_mesh(graph, carved);
  const auto from_selection =
      tf::cpp::make_csg_mesh(graph, tf::csg::selection(carved));
  CHECK(from_expression.size() > 0);
  CHECK(arrays_equal(tf::cpp::test::face_indices_of(from_expression),
                     tf::cpp::test::face_indices_of(from_selection)));
  CHECK(arrays_equal(result_points(from_expression),
                     result_points(from_selection)));

  CHECK_THROWS_AS(
      tf::cpp::make_csg_mesh(graph, tf::csg::selection({2}, tf::csg::op(0))),
      std::out_of_range);
}

TEMPLATE_TEST_CASE("csg graph domains take a boundary selection only",
                   "[cpp][csg][graph][domains][selection]", float, double) {
  const auto owners = boxes<TestType>();
  const auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
  const auto overlap = tf::csg::op(0) & tf::csg::op(1);
  const auto config = tf::domain_config::exclude_outer_shell |
                      tf::domain_config::ignore_open_fragments;

  const auto from_expression =
      tf::cpp::make_csg_domains(graph, overlap, config);
  const auto from_selection =
      tf::cpp::make_csg_domains(graph, tf::csg::selection(overlap), config);
  const auto restricted = tf::cpp::make_csg_domains(
      graph, tf::csg::selection({0}, overlap), config);
  const auto labeled = tf::cpp::make_csg_domains_with_labels(
      graph, tf::csg::selection({0}, overlap), config);
  const auto mapped = tf::cpp::make_csg_domains_with_index_map(
      graph, tf::csg::selection({0}, overlap), config);

  REQUIRE(from_expression.meshes.size() == 1);
  REQUIRE(from_selection.meshes.size() == 1);
  CHECK(arrays_equal(from_expression.ids, from_selection.ids));
  CHECK(arrays_equal(tf::cpp::test::face_indices_of(from_expression.meshes[0]),
                     tf::cpp::test::face_indices_of(from_selection.meshes[0])));

  // the overlap cell is walled by both boxes, so naming one keeps its walls
  REQUIRE(restricted.meshes.size() == 1);
  CHECK(restricted.meshes[0].size() > 0);
  CHECK(restricted.meshes[0].size() < from_selection.meshes[0].size());
  CHECK(labeled.meshes.size() == restricted.meshes.size());
  CHECK(labeled.tag_data.length() > 0);
  for (const auto tag : labeled.tag_data)
    CHECK(tag == tf::cpp::default_index_t{0});
  check_domain_index_map_alignment(mapped, restricted.meshes.size(), 2);

  const auto inside_read = tf::csg::inside({0}, overlap);
  CHECK_THROWS_AS(tf::cpp::make_csg_domains(graph, inside_read, config),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::make_csg_domains_with_labels(graph, inside_read, config),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::make_csg_domains_with_index_map(graph, inside_read, config),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_csg_domains(
                      graph, tf::csg::selection({2}, overlap), config),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("csg graph validates construction expressions and queries",
                   "[cpp][csg][graph][validation]", float, double) {
  using Mesh = tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3, 3>;
  const auto owners = boxes<TestType>();
  const auto inputs = meshes_of(owners);
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(std::vector<Mesh>{inputs[0]}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(inputs, {-1}), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(inputs, {2}), std::out_of_range);

  auto bad_config = tf::arrangement_config{};
  bad_config.intersect.tolerance = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(inputs, {}, bad_config),
                  std::invalid_argument);
  bad_config = {};
  bad_config.triangulation = static_cast<tf::triangulation_type>(2);
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(inputs, {}, bad_config),
                  std::invalid_argument);

  // a default-constructed owner is the EMPTY mesh, which bounds nothing, so a
  // graph of one empty operand and one box is a graph of one box
  const csg_owned<tf::cpp::default_index_t, TestType> nothing;
  CHECK(tf::cpp::make_csg_graph(std::vector<Mesh>{nothing.mesh(), inputs[1]})
            .is_valid());

  // a face naming a point the geometry does not have is refused at the one
  // door the reading passes
  csg_owned<tf::cpp::default_index_t, TestType> bad_indices{
      tf::cpp::make_box_mesh(TestType{1}, TestType{1}, TestType{1})};
  bad_indices.polygons.points_buffer().data_buffer().clear();
  bad_indices.cache.points_changed();
  CHECK_THROWS_AS(
      tf::cpp::make_csg_graph(std::vector<Mesh>{bad_indices.mesh(), inputs[1]}),
      std::out_of_range);

  auto graph = tf::cpp::make_csg_graph(inputs);
  CHECK_THROWS_AS(tf::cpp::make_csg_mesh(graph, tf::csg::op(2)),
                  std::out_of_range);
  CHECK_THROWS_AS(
      tf::cpp::make_csg_domains(graph, static_cast<tf::domain_config>(4)),
      std::invalid_argument);
  const auto malformed = tf::csg::expr(tf::csg::expr::kind::complement, {});
  CHECK_THROWS_AS(tf::cpp::make_csg_mesh(graph, malformed),
                  std::invalid_argument);

  tf::cpp::csg_graph<tf::cpp::default_index_t, TestType> invalid_graph;
  CHECK_THROWS_AS(tf::cpp::make_csg_mesh(invalid_graph), std::invalid_argument);
  graph.destroy();
  CHECK_THROWS_AS(tf::cpp::csg_created_points(graph), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "csg graph async borrows operands and resolves each query once",
    "[cpp][csg][graph][async]", float, double) {
  const auto owners = boxes<TestType>();
  auto pending_graph = tf::cpp::async::make_csg_graph(meshes_of(owners));
  static_assert(
      std::is_same_v<
          decltype(pending_graph),
          std::future<tf::cpp::csg_graph<tf::cpp::default_index_t, TestType>>>);
  auto graph = pending_graph.get();

  const auto expression = tf::csg::op(0) | tf::csg::op(1);
  auto pending_mesh = tf::cpp::async::make_csg_mesh(graph, expression);
  static_assert(
      std::is_same_v<decltype(pending_mesh),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  CHECK(pending_mesh.get().size() > 0);

  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::make_csg_domains(
      counting_resolver{submissions}, graph, expression,
      tf::domain_config::exclude_outer_shell);
  static_assert(std::is_same_v<decltype(custom),
                               std::future<tf::cpp::csg_domains_result<
                                   tf::cpp::default_index_t, TestType>>>);
  CHECK(custom.get().meshes.size() > 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);

  auto failed = tf::cpp::async::make_csg_mesh(graph, tf::csg::op(5));
  CHECK_THROWS_AS(failed.get(), std::out_of_range);
}

TEMPLATE_TEST_CASE("typed csg graph matches the Python dtype matrix",
                   "[cpp][csg][graph][matrix]", float_int32, float_int64,
                   double_int32, double_int64) {
  using Index = typename TestType::index_type;
  using Real = typename TestType::real_type;
  using Mesh = typename TestType::view_type;
  using Graph = typename TestType::graph_type;

  const auto owners = matrix_boxes<TestType>();
  const auto inputs = meshes_of(owners);
  auto graph = tf::cpp::make_csg_graph(inputs);
  static_assert(std::is_same_v<decltype(graph), Graph>);

  const auto build_counts = std::array<std::uint64_t, 4>{
      owners[0].cache.tree_build_count(),
      owners[0].cache.face_membership_build_count(),
      owners[1].cache.tree_build_count(),
      owners[1].cache.face_membership_build_count()};
  const auto union_expression = tf::csg::op(0) | tf::csg::op(1);
  const auto intersection_expression = tf::csg::op(0) & tf::csg::op(1);
  const auto difference_expression = tf::csg::op(0) - tf::csg::op(1);
  const auto false_expression = tf::csg::op(0) & ~tf::csg::op(0);

  const auto full = tf::cpp::make_csg_mesh(graph);
  const auto full_labeled = tf::cpp::make_csg_mesh_with_labels(graph);
  const auto merged = tf::cpp::make_csg_mesh(graph, union_expression);
  const auto repeated = tf::cpp::make_csg_mesh(graph, union_expression);
  const auto intersection =
      tf::cpp::make_csg_mesh(graph, intersection_expression);
  const auto difference = tf::cpp::make_csg_mesh(graph, difference_expression);
  static_assert(std::is_same_v<decltype(merged),
                               const tf::polygons_buffer<Index, Real, 3, 3>>);
  CHECK(full.size() > merged.size());
  CHECK(full_labeled.mesh.size() == full.size());
  CHECK(full_labeled.tag_labels.length() == full.size());
  CHECK(full_labeled.face_labels.length() == full_labeled.tag_labels.length());
  CHECK(arrays_equal(tf::cpp::test::face_indices_of(merged),
                     tf::cpp::test::face_indices_of(repeated)));
  CHECK(arrays_equal(result_points(merged), result_points(repeated)));
  CHECK(static_cast<double>(volume(merged)) ==
        Catch::Approx(1.664).margin(1e-5));
  CHECK(static_cast<double>(volume(intersection)) ==
        Catch::Approx(0.336).margin(1e-5));
  CHECK(static_cast<double>(volume(difference)) ==
        Catch::Approx(0.664).margin(1e-5));

  const auto labeled =
      tf::cpp::make_csg_mesh_with_labels(graph, union_expression);
  static_assert(
      std::is_same_v<decltype(labeled.tag_labels), tf::cpp::nd_array<Index>>);
  REQUIRE(labeled.tag_labels.length() == labeled.mesh.size());
  REQUIRE(labeled.face_labels.length() == labeled.tag_labels.length());
  auto saw_first = false;
  auto saw_second = false;
  for (std::size_t face = 0; face < labeled.tag_labels.length(); ++face) {
    const auto tag = labeled.tag_labels[face];
    const auto source_face = labeled.face_labels[face];
    REQUIRE((tag == Index{0} || tag == Index{1}));
    CHECK(source_face >= Index{0});
    CHECK(source_face <
          static_cast<Index>(
              owners[static_cast<std::size_t>(tag)].polygons.size()));
    saw_first |= tag == Index{0};
    saw_second |= tag == Index{1};
  }
  CHECK(saw_first);
  CHECK(saw_second);

  const auto mapped =
      tf::cpp::make_csg_mesh_with_index_map(graph, union_expression);
  static_assert(
      std::is_same_v<decltype(mapped.point_labels), tf::cpp::nd_array<Index>>);
  CHECK(arrays_equal(mapped.face_tag_labels, labeled.tag_labels));
  CHECK(arrays_equal(mapped.face_labels, labeled.face_labels));
  CHECK(mapped.number_of_tags == Index{2});
  REQUIRE(mapped.point_f_offsets.length() == 3);
  CHECK(mapped.point_f_offsets[0] == Index{0});
  CHECK(mapped.point_f_offsets[2] ==
        static_cast<Index>(mapped.point_f_data.length()));

  const auto created = tf::cpp::csg_created_points(graph);
  const auto curves = tf::cpp::csg_intersection_curves(graph);
  static_assert(std::is_same_v<decltype(tf::cpp::test::arrays_of(curves).ids),
                               tf::cpp::nd_array<Index>>);
  CHECK(created.raw_shape() ==
        tf::small_vector<int, 3>{created.shape_at(0), 3});
  CHECK(created.shape_at(0) > 0);
  CHECK(curves.size() > 0);
  for (const auto point : tf::cpp::test::arrays_of(curves).ids) {
    CHECK(point >= Index{0});
    CHECK(point < static_cast<Index>(
                      tf::cpp::test::arrays_of(curves).points.shape_at(0)));
  }

  const auto domains = tf::cpp::make_csg_domains(
      graph, tf::domain_config::exclude_outer_shell |
                 tf::domain_config::ignore_open_fragments);
  const auto domain_labels = tf::cpp::make_csg_domains_with_labels(
      graph, tf::domain_config::exclude_outer_shell |
                 tf::domain_config::ignore_open_fragments);
  const auto domain_map = tf::cpp::make_csg_domains_with_index_map(
      graph, tf::domain_config::exclude_outer_shell |
                 tf::domain_config::ignore_open_fragments);
  const auto selected_domains =
      tf::cpp::make_csg_domains(graph, intersection_expression,
                                tf::domain_config::exclude_outer_shell |
                                    tf::domain_config::ignore_open_fragments);
  const auto selected_labels = tf::cpp::make_csg_domains_with_labels(
      graph, intersection_expression,
      tf::domain_config::exclude_outer_shell |
          tf::domain_config::ignore_open_fragments);
  const auto selected_map = tf::cpp::make_csg_domains_with_index_map(
      graph, intersection_expression,
      tf::domain_config::exclude_outer_shell |
          tf::domain_config::ignore_open_fragments);
  REQUIRE(domains.meshes.size() == 3);
  CHECK(arrays_equal(domains.ids, domain_labels.ids));
  CHECK(arrays_equal(domains.ids, domain_map.ids));
  REQUIRE(domain_labels.tag_offsets.length() == domains.meshes.size() + 1);
  REQUIRE(domain_labels.face_offsets.length() == domains.meshes.size() + 1);
  CHECK(domain_labels.tag_offsets[0] == Index{0});
  CHECK(domain_labels.face_offsets[0] == Index{0});
  CHECK(domain_map.inclusion.raw_shape() == tf::small_vector<int, 3>{3, 2});
  for (const auto inclusion : domain_map.inclusion)
    CHECK((inclusion == std::int8_t{0} || inclusion == std::int8_t{1}));
  REQUIRE(selected_domains.meshes.size() == 1);
  CHECK(arrays_equal(selected_domains.ids, selected_labels.ids));
  CHECK(arrays_equal(selected_domains.ids, selected_map.ids));
  REQUIRE(selected_map.inclusion.length() == 2);
  CHECK(selected_map.inclusion[0] == std::int8_t{1});
  CHECK(selected_map.inclusion[1] == std::int8_t{1});

  const auto empty_mesh =
      tf::cpp::make_csg_mesh_with_labels(graph, false_expression);
  CHECK(empty_mesh.mesh.size() == 0);
  CHECK(empty_mesh.tag_labels.empty());
  CHECK(empty_mesh.face_labels.empty());
  const auto empty_domains = tf::cpp::make_csg_domains_with_index_map(
      graph, false_expression,
      tf::domain_config::exclude_outer_shell |
          tf::domain_config::ignore_open_fragments);
  CHECK(empty_domains.meshes.empty());
  CHECK(empty_domains.ids.empty());
  CHECK(empty_domains.inclusion.raw_shape() == tf::small_vector<int, 3>{0, 2});
  const auto check_empty_blocks = [](const auto &offsets, const auto &data) {
    CHECK(data.empty());
    REQUIRE(offsets.length() <= 1);
    if (!offsets.empty())
      CHECK(offsets[0] == Index{0});
  };
  check_empty_blocks(empty_domains.face_tag_offsets,
                     empty_domains.face_tag_data);
  check_empty_blocks(empty_domains.face_offsets, empty_domains.face_data);
  check_empty_blocks(empty_domains.point_tag_offsets,
                     empty_domains.point_tag_data);
  check_empty_blocks(empty_domains.point_offsets, empty_domains.point_data);

  CHECK(owners[0].cache.tree_build_count() == build_counts[0]);
  CHECK(owners[0].cache.face_membership_build_count() == build_counts[1]);
  CHECK(owners[1].cache.tree_build_count() == build_counts[2]);
  CHECK(owners[1].cache.face_membership_build_count() == build_counts[3]);

  CHECK_THROWS_AS(tf::cpp::make_csg_graph(std::vector<Mesh>{inputs[0]}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_csg_graph(inputs, {2}), std::out_of_range);
  // a csg arrangement is triangles, so the operand type says so and a mixed
  // owner is refused where it is written, not where it is read
  static_assert(csg_graph_takes<std::vector<Mesh>>::value);
  static_assert(
      !csg_graph_takes<
          std::vector<tf::cpp::mesh<Index, Real, 3, tf::dynamic_size>>>::value);
  csg_owned<Index, Real> bad_indices{
      tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1})};
  bad_indices.polygons.points_buffer().data_buffer().clear();
  bad_indices.cache.points_changed();
  CHECK_THROWS_AS(
      tf::cpp::make_csg_graph(std::vector<Mesh>{bad_indices.mesh(), inputs[1]}),
      std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::make_csg_mesh(graph, tf::csg::op(2)),
                  std::out_of_range);
  Graph invalid;
  CHECK_THROWS_AS(tf::cpp::make_csg_mesh(invalid), std::invalid_argument);
}

TEMPLATE_TEST_CASE("typed csg graph async answers from the graph it built",
                   "[cpp][csg][graph][matrix][async]", float_int64,
                   double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Graph = typename TestType::graph_type;

  const auto owners = matrix_boxes<TestType>();
  auto pending_graph = tf::cpp::async::make_csg_graph(meshes_of(owners));
  static_assert(std::is_same_v<decltype(pending_graph), std::future<Graph>>);
  auto graph = pending_graph.get();

  const auto expression = tf::csg::op(0) | tf::csg::op(1);
  auto pending_mesh = tf::cpp::async::make_csg_mesh(graph, expression);
  static_assert(
      std::is_same_v<decltype(pending_mesh),
                     std::future<tf::polygons_buffer<Index, Real, 3, 3>>>);
  graph.destroy();
  CHECK(pending_mesh.get().size() > 0);

  const auto second_owners = matrix_boxes<TestType>();
  graph = tf::cpp::async::make_csg_graph(meshes_of(second_owners)).get();
  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending_labels = tf::cpp::async::make_csg_mesh_with_labels(
      counting_resolver{submissions}, graph, expression);
  static_assert(std::is_same_v<
                decltype(pending_labels),
                std::future<tf::cpp::csg_mesh_labeled_result<Index, Real>>>);
  graph.destroy();
  CHECK(pending_labels.get().mesh.size() > 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
}
