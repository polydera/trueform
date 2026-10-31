/**
 * @file test_nth_element.cpp
 * @brief Tests for tf::nth_element against std::nth_element.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <functional>
#include <random>
#include <trueform/core.hpp>
#include <vector>

namespace {

struct nth_keyed {
  int key;
  int tag;
};

struct nth_keyed_less {
  auto operator()(const nth_keyed &a, const nth_keyed &b) const -> bool {
    return a.key < b.key;
  }
};

template <typename T, typename Less>
auto nth_check(std::vector<T> input, std::size_t nth, Less less) -> void {
  auto expected = input;
  std::nth_element(expected.begin(), expected.begin() + std::ptrdiff_t(nth),
                   expected.end(), less);
  tf::nth_element(input.begin(), input.begin() + std::ptrdiff_t(nth),
                  input.end(), less);
  if (nth == input.size())
    return;
  REQUIRE(!less(input[nth], expected[nth]));
  REQUIRE(!less(expected[nth], input[nth]));
  for (std::size_t i = 0; i < nth; ++i)
    REQUIRE(!less(input[nth], input[i]));
  for (std::size_t i = nth + 1; i < input.size(); ++i)
    REQUIRE(!less(input[i], input[nth]));
}

template <typename T, typename Less, typename Make>
auto nth_check_shapes(std::size_t n, Less less, Make make) -> void {
  std::mt19937 rng(std::uint32_t(n) * 2654435761u + 17u);
  const std::size_t nths[] = {0, n / 4, n / 2, n - n / 4 - (n ? 1 : 0),
                              n ? n - 1 : 0, n};
  std::vector<T> random(n), equal(n), sorted(n), reversed(n), few(n);
  for (std::size_t i = 0; i < n; ++i) {
    random[i] = make(int(rng() % 1000003), int(i));
    equal[i] = make(7, int(i));
    sorted[i] = make(int(i), int(i));
    reversed[i] = make(int(n - i), int(i));
    few[i] = make(int(rng() % 5), int(i));
  }
  for (auto nth : nths) {
    nth_check(random, nth, less);
    nth_check(equal, nth, less);
    nth_check(sorted, nth, less);
    nth_check(reversed, nth, less);
    nth_check(few, nth, less);
  }
}

const std::size_t nth_sizes[] = {0,   1,   2,   3,    23,   24,   25,
                                 100, 129, 200, 1000, 5000, 70000};

} // namespace

TEST_CASE("nth_element on arithmetic keys under the default comparator",
          "[core][nth_element]") {
  for (auto n : nth_sizes) {
    nth_check_shapes<int>(n, std::less<int>(),
                          [](int key, int) { return key; });
    nth_check_shapes<double>(n, std::less<double>(),
                             [](int key, int) { return double(key) * 0.5; });
  }
}

TEST_CASE("nth_element under std::greater and the two-argument overload",
          "[core][nth_element]") {
  for (auto n : nth_sizes)
    nth_check_shapes<int>(n, std::greater<int>(),
                          [](int key, int) { return key; });
  std::mt19937 rng(5);
  std::vector<int> v(3000);
  for (auto &x : v)
    x = int(rng() % 997);
  auto expected = v;
  std::nth_element(expected.begin(), expected.begin() + 1500, expected.end());
  tf::nth_element(v.begin(), v.begin() + 1500, v.end());
  REQUIRE(v[1500] == expected[1500]);
}

TEST_CASE("nth_element on structs under a custom comparator",
          "[core][nth_element]") {
  for (auto n : nth_sizes)
    nth_check_shapes<nth_keyed>(
        n, nth_keyed_less(),
        [](int key, int tag) { return nth_keyed{key, tag}; });
}
