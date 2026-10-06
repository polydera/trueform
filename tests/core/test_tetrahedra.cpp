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
#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <trueform/core.hpp>
#include <type_traits>
#include <utility>

namespace {
template <typename T> constexpr bool tetra_test_is_primitive = false;
template <typename Policy>
constexpr bool tetra_test_is_primitive<tf::tetrahedron<Policy>> = true;
template <typename T> constexpr bool tetra_test_is_connectivity = false;
template <typename Policy>
constexpr bool tetra_test_is_connectivity<tf::tetras<Policy>> = true;
template <typename T> constexpr bool tetra_test_is_form = false;
template <typename Policy>
constexpr bool tetra_test_is_form<tf::tetrahedra<Policy>> = true;
} // namespace

TEST_CASE("tetras borrow flat and blocked vertex indices",
          "[core][tetrahedra]") {
  std::array<int, 8> ids{0, 1, 2, 3, 3, 2, 1, 4};
  auto flat = tf::make_tetras(ids);
  auto blocked = tf::make_tetras(tf::make_blocked_range<4>(ids));
  STATIC_REQUIRE(tetra_test_is_connectivity<decltype(flat)>);
  STATIC_REQUIRE(tf::static_size_v<decltype(flat[0])> == 4);
  REQUIRE(flat.size() == 2);
  flat[1][3] = 5;
  REQUIRE(blocked[1][3] == 5);
  REQUIRE(&flat[1][0] == &ids[4]);
  auto readonly = tf::make_tetras(std::as_const(ids));
  STATIC_REQUIRE(
      std::is_const_v<std::remove_reference_t<decltype(readonly[0][0])>>);
  STATIC_REQUIRE(!tf::has_tetras<int>);
  auto mutable_view = tf::make_view(flat);
  STATIC_REQUIRE(tetra_test_is_connectivity<decltype(mutable_view)>);
  auto temporary_view = tf::make_view(tf::make_tetras(ids));
  STATIC_REQUIRE(tetra_test_is_connectivity<decltype(temporary_view)>);
  auto copy = tf::make_tetras(flat);
  STATIC_REQUIRE(std::is_same_v<decltype(copy), decltype(flat)>);
  auto view = tf::make_view(std::as_const(flat));
  STATIC_REQUIRE(tetra_test_is_connectivity<decltype(view)>);
  auto tagged = flat | tf::tag_id(9);
  STATIC_REQUIRE(tetra_test_is_connectivity<decltype(tagged)>);
  REQUIRE(tagged.id() == 9);
  REQUIRE(tagged[1][3] == 5);
}

TEMPLATE_TEST_CASE("tetrahedron owns or borrows four points through its policy",
                   "[core][tetrahedra]", float, double) {
  using point_t = tf::point<TestType, 3>;
  std::array<point_t, 4> points{point_t{0, 0, 0}, point_t{1, 0, 0},
                                point_t{0, 1, 0}, point_t{0, 0, 1}};
  auto owned = tf::make_tetrahedron(points);
  auto borrowed = tf::make_tetrahedron(tf::make_range<4>(points.begin()));
  auto dynamic =
      tf::make_tetrahedron(tf::make_range(points.begin(), points.end()));
  auto &[p0, p1, p2, p3] = dynamic;
  REQUIRE(&p0 == &points[0]);
  REQUIRE(&p1 == &points[1]);
  REQUIRE(&p2 == &points[2]);
  REQUIRE(&p3 == &points[3]);
  STATIC_REQUIRE(tetra_test_is_primitive<decltype(owned)>);
  STATIC_REQUIRE(tf::static_size_v<decltype(owned)> == 4);
  STATIC_REQUIRE(tf::coordinate_dims_v<decltype(owned)> == 3);
  STATIC_REQUIRE(
      std::is_same_v<tf::coordinate_type<decltype(owned)>, TestType>);
  auto &[a, b, c, d] = owned;
  b[0] = 7;
  REQUIRE(owned[1][0] == 7);
  REQUIRE(points[1][0] == 1);
  tf::get<1>(borrowed)[0] = 3;
  REQUIRE(points[1][0] == 3);
  borrowed = owned;
  REQUIRE(points[1][0] == 7);
  REQUIRE(&a == &owned[0]);
  REQUIRE(c[1] == 1);
  REQUIRE(d[2] == 1);
  const auto readonly =
      tf::make_tetrahedron(tf::make_range<4>(points.cbegin()));
  STATIC_REQUIRE(
      std::is_const_v<std::remove_reference_t<decltype(readonly[0][0])>>);
}

TEMPLATE_TEST_CASE("tetrahedra preserve indices and compose policies",
                   "[core][tetrahedra]", float, double) {
  using point_t = tf::point<TestType, 3>;
  std::array<point_t, 5> points{point_t{0, 0, 0}, point_t{1, 0, 0},
                                point_t{0, 1, 0}, point_t{0, 0, 1},
                                point_t{0, 0, -1}};
  std::array<std::array<int, 4>, 2> ids{{{0, 1, 2, 3}, {0, 2, 1, 4}}};
  auto tetrahedra = tf::make_tetrahedra(ids, points);
  STATIC_REQUIRE(tetra_test_is_form<decltype(tetrahedra)>);
  STATIC_REQUIRE(tf::has_tetras<decltype(tetrahedra)>);
  STATIC_REQUIRE(tetra_test_is_primitive<decltype(tetrahedra[0])>);
  REQUIRE(tetrahedra.size() == 2);
  REQUIRE(tetrahedra.back()[3][2] == -1);
  REQUIRE(tetrahedra[1].indices()[1] == 2);
  tetrahedra[0][1][0] = 2;
  REQUIRE(tetrahedra[1][2][0] == 2);
  auto tagged =
      tetrahedra | tf::tag_id(42) |
      tf::tag_frame(tf::make_frame(tf::make_transformation_from_translation(
          tf::make_vector(TestType(1), TestType(2), TestType(3)))));
  STATIC_REQUIRE(tetra_test_is_form<decltype(tagged)>);
  STATIC_REQUIRE(tf::has_frame_policy<decltype(tagged)>);
  STATIC_REQUIRE(tf::has_tetras<decltype(tagged)>);
  REQUIRE(tagged.id() == 42);
  REQUIRE(tagged.tetras()[1][3] == 4);
  auto identity = tf::make_tetrahedra(tagged);
  STATIC_REQUIRE(std::is_same_v<decltype(identity), decltype(tagged)>);
  REQUIRE(identity.id() == 42);
  auto view = tf::make_view(tagged);
  STATIC_REQUIRE(std::is_same_v<decltype(view), decltype(tagged)>);
  REQUIRE(view.id() == 42);
  auto readonly =
      tf::make_tetrahedra(std::as_const(ids), std::as_const(points));
  STATIC_REQUIRE(
      std::is_const_v<std::remove_reference_t<decltype(readonly[0][0][0])>>);
}

TEST_CASE("tetrahedra soup and transformations preserve primitive semantics",
          "[core][tetrahedra]") {
  using point_t = tf::point<double, 3>;
  auto tetrahedron = tf::make_tetrahedron(std::array<point_t, 4>{
      point_t{0, 0, 0}, point_t{1, 0, 0}, point_t{0, 1, 0}, point_t{0, 0, 1}});
  std::array<decltype(tetrahedron), 2> soup{tetrahedron, tetrahedron};
  auto tetrahedra = tf::make_tetrahedra(soup);
  STATIC_REQUIRE(tetra_test_is_form<decltype(tetrahedra)>);
  STATIC_REQUIRE(!tf::has_tetras<decltype(tetrahedra)>);
  REQUIRE(tetrahedra.size() == 2);
  auto transformation =
      tf::make_transformation_from_translation(tf::make_vector(3., 4., 5.));
  auto moved = tf::transformed(tetrahedron | tf::tag_id(7), transformation);
  STATIC_REQUIRE(tetra_test_is_primitive<decltype(moved)>);
  REQUIRE(moved.id() == 7);
  REQUIRE(moved[3][2] == 6);
  std::array<int, 4> ids{3, 2, 1, 0};
  auto indexed = tf::make_tetrahedron(ids, tetrahedron);
  auto framed = tf::transformed(indexed, tf::make_frame(transformation));
  STATIC_REQUIRE(tetra_test_is_primitive<decltype(framed)>);
  REQUIRE(framed.indices()[0] == 3);
  REQUIRE(framed[0][2] == 6);
  REQUIRE(tetrahedron[3][2] == 1);
  auto identity = tf::transformed(indexed, tf::identity_frame<double, 3>{});
  STATIC_REQUIRE(std::is_same_v<decltype(identity), decltype(indexed)>);
}

TEMPLATE_TEST_CASE("tetrahedra buffer materialization separates ownership",
                   "[core][tetrahedra]", std::int32_t, std::int64_t) {
  using point_t = tf::point<double, 3>;
  std::array<point_t, 5> points{point_t{0, 0, 0}, point_t{1, 0, 0},
                                point_t{0, 1, 0}, point_t{0, 0, 1},
                                point_t{7, 8, 9}};
  std::array<TestType, 4> ids{0, 1, 2, 3};
  auto buffer = tf::make_tetrahedra_buffer(tf::make_tetrahedra(ids, points));
  STATIC_REQUIRE(std::is_same_v<decltype(buffer),
                                tf::tetrahedra_buffer<TestType, double>>);
  STATIC_REQUIRE(tf::coordinate_dims_v<decltype(buffer)> == 3);
  STATIC_REQUIRE(std::is_same_v<decltype(buffer.points_buffer()),
                                tf::points_buffer<double, 3> &>);
  REQUIRE(buffer.size() == 1);
  REQUIRE(buffer.points().size() == 5);
  REQUIRE(buffer.points()[4][2] == 9);
  buffer.front()[1][0] = 6;
  buffer.tetras()[0][0] = 4;
  REQUIRE(points[1][0] == 1);
  REQUIRE(ids[0] == 0);
  REQUIRE(buffer.tetrahedra()[0][0][2] == 9);
  auto readonly = std::as_const(buffer).tetrahedra();
  STATIC_REQUIRE(
      std::is_const_v<std::remove_reference_t<decltype(readonly[0][0][0])>>);
  STATIC_REQUIRE(std::is_const_v<
                 std::remove_reference_t<decltype(readonly.tetras()[0][0])>>);
  auto *point_data = buffer.points_buffer().data_buffer().data();
  auto *tetra_data = buffer.tetras_buffer().data_buffer().data();
  auto moved = tf::make_tetrahedra_buffer(std::move(buffer.tetras_buffer()),
                                          std::move(buffer.points_buffer()));
  REQUIRE(moved.points_buffer().data_buffer().data() == point_data);
  REQUIRE(moved.tetras_buffer().data_buffer().data() == tetra_data);
  moved.clear();
  REQUIRE(moved.empty());
  REQUIRE(moved.points().empty());
  REQUIRE(moved.points_buffer().data_buffer().data() == point_data);
  REQUIRE(moved.tetras_buffer().data_buffer().data() == tetra_data);
}

TEST_CASE("empty tetrahedral connectivity retains supplied points when copied",
          "[core][tetrahedra]") {
  std::array<int, 0> ids{};
  std::array<tf::point<double, 3>, 1> points{tf::point<double, 3>{1, 2, 3}};
  auto buffer = tf::make_tetrahedra_buffer(tf::make_tetrahedra(ids, points));
  REQUIRE(buffer.empty());
  REQUIRE(buffer.points().size() == 1);
  REQUIRE(buffer.points()[0][2] == 3);
  tf::tetrahedra_buffer<int, double> empty;
  REQUIRE(empty.empty());
  REQUIRE(empty.begin() == empty.end());
}

TEST_CASE("tetrahedra transformations visit each stored point once",
          "[core][tetrahedra]") {
  using point_t = tf::point<double, 3>;
  std::array<point_t, 5> points{point_t{0, 0, 0}, point_t{1, 0, 0},
                                point_t{0, 1, 0}, point_t{0, 0, 1},
                                point_t{0, 0, -1}};
  std::array<int, 8> ids{0, 1, 2, 3, 0, 2, 1, 4};
  auto tetrahedra = tf::make_tetrahedra(ids, points);
  auto delta = tf::make_vector(3., 4., 5.);
  auto t = tf::make_transformation_from_translation(delta);
  auto copied = tf::transformed(tetrahedra, t);
  auto framed = tf::transformed(tetrahedra, tf::make_frame(t));
  auto translated = tf::translated(tetrahedra, delta);
  STATIC_REQUIRE(
      std::is_same_v<decltype(copied), tf::tetrahedra_buffer<int, double>>);
  REQUIRE(points[0][0] == 0);
  REQUIRE(copied.points()[0][0] == 3);
  REQUIRE(framed.points()[0][1] == 4);
  REQUIRE(translated.points()[0][2] == 5);
  REQUIRE(copied.tetras()[1][2] == 1);
  tf::transform(tetrahedra, t);
  REQUIRE(points[0][0] == 3);
  tf::transform(tf::make_tetrahedra(ids, points), tf::make_frame(t));
  REQUIRE(points[0][0] == 6);
  tf::translate(tetrahedra, delta);
  REQUIRE(points[0][0] == 9);
  tf::translate(tf::make_tetrahedra(ids, points), delta);
  REQUIRE(points[0][0] == 12);
}
