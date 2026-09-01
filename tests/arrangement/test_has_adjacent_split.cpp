/**
 * @file test_has_adjacent_split.cpp
 * @brief tf::arrangement::has_adjacent_split over the two tables it spans.
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
#include <trueform/arrangement/planes/has_adjacent_split.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>

#include <array>

namespace {

using adjacent_split_index_t = int;
using key_t = std::array<adjacent_split_index_t, 2>;
using edge_t = std::array<key_t, 2>;

/// The caller's currency: an edge key plus the position on it.
template <typename Int> struct split_t {
  edge_t edge;
  typename tf::exact::meta<Int>::param_type param;
};

auto edge_of = [](const auto &split) -> const edge_t & { return split.edge; };
auto param_of = [](const auto &split) { return split.param; };

auto edge_key(const key_t &a, const key_t &b) -> edge_t {
  return a < b ? edge_t{a, b} : edge_t{b, a};
}

template <typename Int, typename Param>
auto has_adjacent(const edge_t &edge, Param parameter,
                  const tf::buffer<split_t<Int>> &sorted,
                  const tf::buffer<split_t<Int>> &fresh) -> bool {
  return tf::arrangement::has_adjacent_split(edge, parameter, sorted, fresh,
                                             edge_of, param_of);
}

template <typename Int, typename Param>
auto has_adjacent(const edge_t &edge, Param parameter,
                  const tf::buffer<split_t<Int>> &sorted) -> bool {
  return tf::arrangement::has_adjacent_split(edge, parameter, sorted, edge_of,
                                     param_of);
}

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
  return split_t<Int>{edge, param};
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

TEMPLATE_TEST_CASE("has_adjacent_split: the ordered table answers by "
                   "edge key, not by parameter alone",
                   "[has_adjacent_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = edge_key(v0, v1);
  const auto middle = edge_key(v1, v2);
  const auto high = edge_key(v2, v3);
  const auto absent = edge_key(v0, v3);
  const auto sorted = filled<Int>({make_split<Int>(low, param_t(1000)),
                                   make_split<Int>(middle, param_t(1000)),
                                   make_split<Int>(high, param_t(1000))});
  const tf::buffer<split_t<Int>> fresh;

  CHECK(has_adjacent<Int>(middle, param_t(1000), sorted,
                                                  fresh));
  CHECK(has_adjacent<Int>(middle, param_t(999), sorted,
                                                  fresh));
  CHECK(has_adjacent<Int>(middle, param_t(1001), sorted,
                                                  fresh));
  CHECK_FALSE(has_adjacent<Int>(middle, param_t(998),
                                                        sorted, fresh));
  CHECK_FALSE(has_adjacent<Int>(middle, param_t(1002),
                                                        sorted, fresh));

  // An identical parameter on another edge is another question entirely.
  CHECK_FALSE(has_adjacent<Int>(absent, param_t(1000),
                                                        sorted, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_split: the wave's appends answer the "
                   "same way as the ordered table",
                   "[has_adjacent_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = edge_key(v0, v1);
  const auto middle = edge_key(v1, v2);
  const tf::buffer<split_t<Int>> sorted;
  const auto fresh = filled<Int>({make_split<Int>(middle, param_t(1000))});

  CHECK(has_adjacent<Int>(middle, param_t(1000), sorted,
                                                  fresh));
  CHECK(has_adjacent<Int>(middle, param_t(999), sorted,
                                                  fresh));
  CHECK(has_adjacent<Int>(middle, param_t(1001), sorted,
                                                  fresh));
  CHECK_FALSE(has_adjacent<Int>(middle, param_t(1002),
                                                        sorted, fresh));
  CHECK_FALSE(has_adjacent<Int>(low, param_t(1000),
                                                        sorted, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_split: an append whose key sorts "
                   "before the ordered table is still found",
                   "[has_adjacent_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = edge_key(v0, v1);
  const auto middle = edge_key(v1, v2);
  const auto high = edge_key(v2, v3);
  const auto fresh = filled<Int>({make_split<Int>(low, param_t(4000))});

  const auto one_after = filled<Int>({make_split<Int>(high, param_t(1000))});
  CHECK(has_adjacent<Int>(low, param_t(4000), one_after,
                                                  fresh));
  // The ordered table's split is on another edge, so it suppresses nothing
  // here however the two tables are traversed.
  CHECK_FALSE(has_adjacent<Int>(low, param_t(1000),
                                                        one_after, fresh));

  const auto two_after = filled<Int>({make_split<Int>(middle, param_t(1000)),
                                      make_split<Int>(high, param_t(1000))});
  CHECK(has_adjacent<Int>(low, param_t(4000), two_after,
                                                  fresh));
  CHECK(has_adjacent<Int>(low, param_t(3999), two_after,
                                                  fresh));
  CHECK_FALSE(has_adjacent<Int>(low, param_t(2000),
                                                        two_after, fresh));
}

TEMPLATE_TEST_CASE("has_adjacent_split: the one-table overload is the "
                   "two-table question with nothing fresh",
                   "[has_adjacent_split]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto low = edge_key(v0, v1);
  const auto middle = edge_key(v1, v2);
  const auto absent = edge_key(v0, v3);
  const auto sorted = filled<Int>({make_split<Int>(low, param_t(10)),
                                   make_split<Int>(middle, param_t(1000))});
  const tf::buffer<split_t<Int>> fresh;

  const edge_t edges[] = {low, middle, absent};
  const param_t parameters[] = {param_t(0),    param_t(9),    param_t(10),
                                param_t(11),   param_t(12),   param_t(999),
                                param_t(1000), param_t(1001), param_t(1002)};
  for (const auto &edge : edges)
    for (param_t parameter : parameters)
      CHECK(has_adjacent<Int>(edge, parameter, sorted) ==
            has_adjacent<Int>(edge, parameter, sorted,
                                                      fresh));
}
