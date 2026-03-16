/**
 * @file test_ranges.cpp
 * @brief Tests for tf::zip and tf::zip_with
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core.hpp>
#include <algorithm>

// =============================================================================
// zip_with struct
// =============================================================================

struct record {
  int &key;
  float &value;
  int &tag;
};

// =============================================================================
// Tests
// =============================================================================

TEST_CASE("zip sort reorders all buffers consistently", "[core][zip]") {
  std::vector<int> keys = {3, 1, 4, 1, 5};
  std::vector<float> values = {30.f, 10.f, 40.f, 11.f, 50.f};
  std::vector<int> tags = {0, 1, 2, 3, 4};

  auto zipped = tf::zip(tf::make_range(keys), tf::make_range(values),
                        tf::make_range(tags));

  std::sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  REQUIRE(keys[0] == 1);
  REQUIRE(keys[1] == 1);
  REQUIRE(keys[2] == 3);
  REQUIRE(keys[3] == 4);
  REQUIRE(keys[4] == 5);

  // values follow their keys
  REQUIRE(values[2] == 30.f);
  REQUIRE(values[3] == 40.f);
  REQUIRE(values[4] == 50.f);

  // tags follow their keys
  REQUIRE(tags[2] == 0);
  REQUIRE(tags[3] == 2);
  REQUIRE(tags[4] == 4);
}

TEST_CASE("zip_with iterates with named member access", "[core][zip_with]") {
  std::vector<int> keys = {3, 1, 4, 1, 5};
  std::vector<float> values = {30.f, 10.f, 40.f, 11.f, 50.f};
  std::vector<int> tags = {0, 1, 2, 3, 4};

  auto view = tf::zip_with<record>(tf::make_range(keys), tf::make_range(values),
                                   tf::make_range(tags));

  float sum = 0.f;
  int count = 0;
  for (auto r : view) {
    if (r.key > 2) {
      sum += r.value;
      count += r.tag;
    }
  }

  // keys > 2: {3, 4, 5} -> values {30, 40, 50}, tags {0, 2, 4}
  REQUIRE(sum == 120.f);
  REQUIRE(count == 6);
}

TEST_CASE("zip_with provides named member access", "[core][zip_with]") {
  std::vector<int> keys = {10, 20, 30};
  std::vector<float> values = {1.f, 2.f, 3.f};
  std::vector<int> tags = {100, 200, 300};

  auto view = tf::zip_with<record>(tf::make_range(keys), tf::make_range(values),
                                   tf::make_range(tags));

  REQUIRE(view[0].key == 10);
  REQUIRE(view[0].value == 1.f);
  REQUIRE(view[0].tag == 100);
  REQUIRE(view[2].key == 30);
  REQUIRE(view[2].value == 3.f);
  REQUIRE(view[2].tag == 300);
}
