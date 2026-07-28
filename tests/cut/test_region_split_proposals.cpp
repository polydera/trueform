/**
 * @file test_region_split_proposals.cpp
 * @brief tf::cut::materialize_region_split_proposals — sameness is asked in
 *        the units the answer materializes in.
 *
 * The producer quantises every proposal onto the edge's own grid before it
 * dedups, so two positions closer than one step are one parameter and one
 * created id, and a position the quantisation puts on an end of the edge is
 * dropped instead of becoming a second identity on a vertex that exists.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/cut/detail/region_triangulation_types.hpp>
#include <trueform/cut/impl/materialize_region_split_proposals.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/snap_parameter_to_edge.hpp>
#include <trueform/intersect/graph/vertex.hpp>

#include <array>
#include <cstddef>

namespace {

using Index = int;

template <typename Int> struct edge_case {
  std::array<tf::point<Int, 3>, 2> points;
  typename tf::exact::meta<Int>::param_type step;
};

/// One original edge on tag 0, spanning half the lattice.
template <typename Int> auto make_edge_case() -> edge_case<Int> {
  using param_t = typename tf::exact::meta<Int>::param_type;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const tf::point<Int, 3> a{Int(0), Int(0), Int(0)};
  const tf::point<Int, 3> b{Int(Int(1) << (coordinate_bits - 1)), Int(0),
                            Int(0)};
  return {{a, b},
          param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b)};
}

template <typename Int>
auto make_proposal(typename tf::exact::meta<Int>::param_type param)
    -> tf::cut::detail::region_split_proposal<Index, Int> {
  tf::cut::detail::region_split_proposal<Index, Int> proposal{};
  for (int end = 0; end < 2; ++end) {
    proposal.edge[std::size_t(end)].source =
        tf::intersect::graph::vertex_source::original;
    proposal.edge[std::size_t(end)].id = Index(end);
  }
  proposal.tag = 0;
  proposal.param = param;
  return proposal;
}

template <typename Int>
auto materialize(
    const edge_case<Int> &edge,
    std::initializer_list<typename tf::exact::meta<Int>::param_type> params,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits)
    -> std::size_t {
  tf::buffer<tf::cut::detail::region_split_proposal<Index, Int>> proposals;
  proposals.reserve(params.size());
  for (auto param : params)
    proposals.push_back(make_proposal<Int>(param));
  const tf::buffer<tf::point<Int, 3>> intersection_points;
  return tf::cut::materialize_region_split_proposals<Index, Int>(
      Index(1), Index(0), proposals, intersection_points,
      [&](Index, Index id) { return edge.points[std::size_t(id)]; },
      extra_points, splits);
}

} // namespace

TEMPLATE_TEST_CASE("region split proposals: two positions inside one step "
                   "become one split",
                   "[region_split_proposals]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto edge = make_edge_case<Int>();
  const param_t base = edge.step * param_t(100);
  tf::buffer<tf::point<Int, 3>> extra_points;
  tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> splits;

  // The two parameters are further apart than the tolerance the dedup
  // applies, and nearer than one step of the grid they are stated on.
  CHECK(materialize<Int>(edge, {base + param_t(1), base + param_t(3)},
                         extra_points, splits) == 1u);
  REQUIRE(splits.size() == 1u);
  CHECK(extra_points.size() == 1u);
  CHECK(splits[0].param == base);
}

TEMPLATE_TEST_CASE("region split proposals: a position the grid puts on an end "
                   "is dropped",
                   "[region_split_proposals]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto edge = make_edge_case<Int>();
  const param_t half_step = edge.step / param_t(2);
  const param_t maximum = param_t(1) << tf::exact::meta<Int>::split_grid_bits;

  tf::buffer<tf::point<Int, 3>> extra_points;
  tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> splits;
  CHECK(materialize<Int>(edge, {half_step - param_t(1)}, extra_points,
                         splits) == 0u);
  CHECK(splits.size() == 0u);
  CHECK(extra_points.size() == 0u);

  CHECK(materialize<Int>(edge, {maximum - half_step}, extra_points, splits) ==
        0u);
  CHECK(splits.size() == 0u);
  CHECK(extra_points.size() == 0u);

  // A step away from the end there is an interior position to split at.
  CHECK(materialize<Int>(edge, {half_step, maximum - half_step - param_t(1)},
                         extra_points, splits) == 2u);
  REQUIRE(splits.size() == 2u);
  CHECK(splits[0].param == edge.step);
  CHECK(splits[1].param == maximum - edge.step);
}
