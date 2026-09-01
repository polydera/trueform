/**
 * @file test_segment_arrangements.cpp
 * @brief Tests for tf::make_segment_arrangements
 *
 * Verifies exact segment splitting at VV, VE, and EE intersections
 * with correct point merging and edge labels, in 2D and 3D.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/core/segments_buffer.hpp>
#include <trueform/arrangement/make_segment_arrangements.hpp>
#include "type_traits.hpp"

template <typename Index, typename RealT, std::size_t Dims>
auto make_test_segments() {
  // Main edge: (0,0[,0]) → (4,0[,0])
  // VV at origin: edges with endpoint at (0,0[,0])
  // VE at (4,0[,0]): edge passing through main edge endpoint
  // EE at (2,0[,0]): X-crossing edges

  tf::segments_buffer<Index, RealT, Dims> sb;
  auto &pts = sb.points_buffer().data_buffer();
  auto &edges = sb.edges_buffer().data_buffer();

  pts.allocate(12 * Dims);
  auto set_pt = [&](int i, auto... coords) {
    RealT vals[] = {RealT(coords)...};
    for (std::size_t d = 0; d < Dims; ++d)
      pts[i * Dims + d] = vals[d];
  };

  if constexpr (Dims == 2) {
    set_pt(0, 0, 0);
    set_pt(1, 4, 0);
    set_pt(2, -1, 1);
    set_pt(3, 0, 0);
    set_pt(4, 0, -1);
    set_pt(5, 0, 0);
    set_pt(6, 4, -1);
    set_pt(7, 4, 1);
    set_pt(8, 1, -1);
    set_pt(9, 3, 1);
    set_pt(10, 1, 1);
    set_pt(11, 3, -1);
  } else {
    set_pt(0, 0, 0, 0);
    set_pt(1, 4, 0, 0);
    set_pt(2, -1, 1, 0.5);
    set_pt(3, 0, 0, 0);
    set_pt(4, 0, -1, -0.5);
    set_pt(5, 0, 0, 0);
    set_pt(6, 4, -1, 0.3);
    set_pt(7, 4, 1, -0.3);
    set_pt(8, 1, -1, 0);
    set_pt(9, 3, 1, 0);
    set_pt(10, 1, 1, 0);
    set_pt(11, 3, -1, 0);
  }

  edges.allocate(6 * 2);
  auto set_edge = [&](int i, Index a, Index b) {
    edges[i * 2 + 0] = a;
    edges[i * 2 + 1] = b;
  };
  set_edge(0, 0, 1);
  set_edge(1, 2, 3);
  set_edge(2, 4, 5);
  set_edge(3, 6, 7);
  set_edge(4, 8, 9);
  set_edge(5, 10, 11);

  return sb;
}

TEMPLATE_TEST_CASE("Segment arrangements 2D: VV + VE + EE",
                   "[segment_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sb = make_test_segments<index_t, real_t, 2>();
  auto [out, labels] = tf::make_segment_arrangements(sb.segments());

  REQUIRE(out.edges().size() == 10);
  REQUIRE(out.points().size() == 11);
  REQUIRE(labels.size() == 10);

  int counts[6] = {};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    REQUIRE(labels[i] >= 0);
    REQUIRE(labels[i] < 6);
    counts[labels[i]]++;
  }
  REQUIRE(counts[0] == 2);
  REQUIRE(counts[1] == 1);
  REQUIRE(counts[2] == 1);
  REQUIRE(counts[3] == 2);
  REQUIRE(counts[4] == 2);
  REQUIRE(counts[5] == 2);
}

TEMPLATE_TEST_CASE("Segment arrangements 3D: VV + VE + EE",
                   "[segment_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto sb = make_test_segments<index_t, real_t, 3>();
  auto [out, labels] = tf::make_segment_arrangements(sb.segments());

  REQUIRE(out.edges().size() == 10);
  REQUIRE(out.points().size() == 11);
  REQUIRE(labels.size() == 10);

  int counts[6] = {};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    REQUIRE(labels[i] >= 0);
    REQUIRE(labels[i] < 6);
    counts[labels[i]]++;
  }
  REQUIRE(counts[0] == 2);
  REQUIRE(counts[1] == 1);
  REQUIRE(counts[2] == 1);
  REQUIRE(counts[3] == 2);
  REQUIRE(counts[4] == 2);
  REQUIRE(counts[5] == 2);
}

TEMPLATE_TEST_CASE("Segment arrangements: no intersections",
                   "[segment_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  tf::segments_buffer<index_t, real_t, 2> sb;
  auto &pts = sb.points_buffer().data_buffer();
  auto &edges = sb.edges_buffer().data_buffer();

  pts.allocate(4 * 2);
  pts[0] = 0; pts[1] = 0;
  pts[2] = 1; pts[3] = 0;
  pts[4] = 2; pts[5] = 1;
  pts[6] = 3; pts[7] = 1;

  edges.allocate(2 * 2);
  edges[0] = 0; edges[1] = 1;
  edges[2] = 2; edges[3] = 3;

  auto [out, labels] = tf::make_segment_arrangements(sb.segments());

  REQUIRE(out.edges().size() == 2);
  REQUIRE(out.points().size() == 4);
  REQUIRE(labels.size() == 2);
  REQUIRE(labels[0] == 0);
  REQUIRE(labels[1] == 1);
}
