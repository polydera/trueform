#include <catch2/catch_test_macros.hpp>

#include <trueform/core/algorithm/compact_ids_and_make_map.hpp>
#include <trueform/core/buffer.hpp>

#include <array>

TEST_CASE("compact_ids_and_make_map compacts in first occurrence order",
          "[core][algorithm]") {
  std::array<int, 5> ids{7, 2, 7, 5, 2};
  auto [count, old_to_new] =
      tf::compact_ids_and_make_map(tf::make_range(ids), int(8));

  CHECK(count == 3);
  CHECK(ids == std::array<int, 5>{0, 1, 0, 2, 1});
  REQUIRE(old_to_new.size() == 8);
  CHECK(old_to_new[2] == 1);
  CHECK(old_to_new[5] == 2);
  CHECK(old_to_new[7] == 0);
  CHECK(old_to_new[0] == 8);
  CHECK(old_to_new[1] == 8);
  CHECK(old_to_new[3] == 8);
  CHECK(old_to_new[4] == 8);
  CHECK(old_to_new[6] == 8);
}

TEST_CASE("compact_ids_and_make_map accepts caller-owned map storage",
          "[core][algorithm]") {
  std::array<int, 6> ids{4, 1, 4, 0, 1, 3};
  tf::buffer<int> old_to_new;
  old_to_new.allocate(6);

  const auto count = tf::compact_ids_and_make_map(tf::make_range(ids),
                                                  tf::make_range(old_to_new));

  CHECK(count == 4);
  CHECK(ids == std::array<int, 6>{0, 1, 0, 2, 1, 3});
  CHECK(old_to_new[4] == 0);
  CHECK(old_to_new[1] == 1);
  CHECK(old_to_new[0] == 2);
  CHECK(old_to_new[3] == 3);
  CHECK(old_to_new[2] == 6);
  CHECK(old_to_new[5] == 6);
}

TEST_CASE("compact_ids_and_make_map handles an empty occurrence stream",
          "[core][algorithm]") {
  std::array<int, 0> ids{};
  auto [count, old_to_new] =
      tf::compact_ids_and_make_map(tf::make_range(ids), int(4));

  CHECK(count == 0);
  REQUIRE(old_to_new.size() == 4);
  for (auto mapped : old_to_new)
    CHECK(mapped == 4);
}
