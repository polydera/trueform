/**
 * @file test_has_adjacent_region_split.cpp
 * @brief tf::cut::has_adjacent_region_split over the two tables it spans.
 *
 * The ordered table answers by binary search, the wave's appends are
 * scanned. Pins that the edge key decides the answer in both of them, that
 * an append whose key sorts before the ordered table is still found, and
 * that the one-table overload is the same question with nothing fresh.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/cut/detail/region_triangulation_identity.hpp>
#include <trueform/cut/detail/region_triangulation_types.hpp>
#include <trueform/cut/impl/has_adjacent_region_split.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>

#include <array>

namespace {

using Index = int;
using key_t = std::array<Index, 2>;
using edge_t = std::array<key_t, 2>;

template <typename Int>
using split_t = tf::cut::detail::region_triangulation_split<Index, Int>;

/// Four vertex keys, so the three edges built from them are strictly
/// ordered: {0,10} < {0,11} < {0,12} < {1,4}.
constexpr key_t v0{0, 10};
constexpr key_t v1{0, 11};
constexpr key_t v2{0, 12};
constexpr key_t v3{1, 4};

template <typename Int>
auto make_split(const edge_t &edge,
                typename tf::exact::meta<Int>::param_type param)
    -> split_t<Int> {
  split_t<Int> split{};
  split.edge = edge;
  split.param = param;
  return split;
}

template <typename Int>
auto filled(std::initializer_list<split_t<Int>> splits)
    -> tf::buffer<split_t<Int>> {
  tf::buffer<split_t<Int>> buffer;
  buffer.reserve(splits.size());
  for (const auto &split : splits)
    buffer.push_back(split);
  return buffer;
}

} // namespace

TEMPLATE_TEST_CASE("has_adjacent_region_split: the ordered table answers by "
                   "edge key, not by parameter alone",
                   "[has_adjacent_region_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = tf::cut::detail::region_edge_key(v0, v1);
  const auto middle = tf::cut::detail::region_edge_key(v1, v2);
  const auto high = tf::cut::detail::region_edge_key(v2, v3);
  const auto absent = tf::cut::detail::region_edge_key(v0, v3);
  const auto sorted = filled<Int>({make_split<Int>(low, param_t(1000)),
                                   make_split<Int>(middle, param_t(1000)),
                                   make_split<Int>(high, param_t(1000))});
  const tf::buffer<split_t<Int>> fresh;

  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1000), sorted,
                                                  fresh));
  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(999), sorted,
                                                  fresh));
  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1001), sorted,
                                                  fresh));
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(middle, param_t(998),
                                                        sorted, fresh));
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1002),
                                                        sorted, fresh));

  // An identical parameter on another edge is another question entirely.
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(absent, param_t(1000),
                                                        sorted, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_region_split: the wave's appends answer the "
                   "same way as the ordered table",
                   "[has_adjacent_region_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = tf::cut::detail::region_edge_key(v0, v1);
  const auto middle = tf::cut::detail::region_edge_key(v1, v2);
  const tf::buffer<split_t<Int>> sorted;
  const auto fresh = filled<Int>({make_split<Int>(middle, param_t(1000))});

  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1000), sorted,
                                                  fresh));
  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(999), sorted,
                                                  fresh));
  CHECK(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1001), sorted,
                                                  fresh));
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(middle, param_t(1002),
                                                        sorted, fresh));
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(low, param_t(1000),
                                                        sorted, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_region_split: an append whose key sorts "
                   "before the ordered table is still found",
                   "[has_adjacent_region_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = tf::cut::detail::region_edge_key(v0, v1);
  const auto middle = tf::cut::detail::region_edge_key(v1, v2);
  const auto high = tf::cut::detail::region_edge_key(v2, v3);
  const auto fresh = filled<Int>({make_split<Int>(low, param_t(4000))});

  const auto one_after = filled<Int>({make_split<Int>(high, param_t(1000))});
  CHECK(tf::cut::has_adjacent_region_split<Index>(low, param_t(4000), one_after,
                                                  fresh));
  // The ordered table's split is on another edge, so it suppresses nothing
  // here however the two tables are traversed.
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(low, param_t(1000),
                                                        one_after, fresh));

  const auto two_after = filled<Int>({make_split<Int>(middle, param_t(1000)),
                                      make_split<Int>(high, param_t(1000))});
  CHECK(tf::cut::has_adjacent_region_split<Index>(low, param_t(4000), two_after,
                                                  fresh));
  CHECK(tf::cut::has_adjacent_region_split<Index>(low, param_t(3999), two_after,
                                                  fresh));
  CHECK_FALSE(tf::cut::has_adjacent_region_split<Index>(low, param_t(2000),
                                                        two_after, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_region_split: the one-table overload is the "
                   "two-table question with nothing fresh",
                   "[has_adjacent_region_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = tf::cut::detail::region_edge_key(v0, v1);
  const auto middle = tf::cut::detail::region_edge_key(v1, v2);
  const auto absent = tf::cut::detail::region_edge_key(v0, v3);
  const auto sorted = filled<Int>({make_split<Int>(low, param_t(10)),
                                   make_split<Int>(middle, param_t(1000))});
  const tf::buffer<split_t<Int>> fresh;

  const edge_t edges[] = {low, middle, absent};
  const param_t parameters[] = {param_t(0),    param_t(9),    param_t(10),
                                param_t(11),   param_t(12),   param_t(999),
                                param_t(1000), param_t(1001), param_t(1002)};
  for (const auto &edge : edges)
    for (param_t parameter : parameters)
      CHECK(tf::cut::has_adjacent_region_split<Index>(edge, parameter, sorted) ==
            tf::cut::has_adjacent_region_split<Index>(edge, parameter, sorted,
                                                      fresh));
}
