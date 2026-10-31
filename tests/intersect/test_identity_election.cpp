/**
 * @file test_identity_election.cpp
 * @brief THE ELECTION over hand-stated coincidences.
 *
 * One coincidence: a standing created identity survives and every other
 * standing one and every original states its rewrite row, a fresh
 * statement states none; a coincidence holding no standing identity mints,
 * in the order the caller states its coincidences. The sparse-relation
 * driver closes the relations into coincidences and elects them at their
 * roots, ascending, whatever order the relations were stated in.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/intersect/graph/elect_identity_components.hpp>

#include <array>
#include <initializer_list>
#include <vector>

namespace {

using election_index_t = int;
using election_key_t = tf::intersect::graph::identity_key<election_index_t>;
using election_relation_t =
    tf::intersect::graph::identity_relation<election_index_t>;
using election_row_t = std::array<election_index_t, 3>;

template <typename T>
auto election_filled(std::initializer_list<T> values) -> tf::buffer<T> {
  tf::buffer<T> buffer;
  buffer.reserve(values.size());
  for (const auto &value : values)
    buffer.push_back(value);
  return buffer;
}

template <typename T>
auto election_rows(const tf::buffer<T> &buffer) -> std::vector<T> {
  return std::vector<T>(buffer.begin(), buffer.end());
}

/// A caller's created space past 100: each mint takes the next number and
/// records the winner that asked for it.
struct election_minter {
  std::vector<election_key_t> *winners;
  auto operator()(const election_key_t &winner) const -> election_index_t {
    winners->push_back(winner);
    return election_index_t(100 + winners->size() - 1);
  }
};

auto elect_one(std::initializer_list<election_key_t> component,
               std::vector<election_key_t> &winners,
               tf::buffer<election_row_t> &merges) -> election_index_t {
  const auto keys = election_filled<election_key_t>(component);
  return tf::intersect::graph::elect_identity_component<election_index_t>(
      tf::make_range(keys), election_minter{&winners}, merges);
}

} // namespace

TEST_CASE("election: a standing created identity survives",
          "[intersect][identity]") {
  std::vector<election_key_t> winners;
  tf::buffer<election_row_t> merges;
  const auto target = elect_one(
      {{0, 5}, {0, 7}, {1, 3}, {1, 4}, {2, 40}, {2, 41}}, winners, merges);
  CHECK(target == 5);
  CHECK(winners.empty());
  // the losing standing identity and the originals state rows; the fresh
  // statements state none and the survivor never states itself
  CHECK(election_rows(merges) ==
        std::vector<election_row_t>{{1, 7, 5}, {0, 40, 5}, {0, 41, 5}});
}

TEST_CASE("election: a coincidence of no standing identity mints",
          "[intersect][identity]") {
  std::vector<election_key_t> winners;
  tf::buffer<election_row_t> merges;
  const auto target = elect_one({{1, 3}, {1, 9}, {2, 4}}, winners, merges);
  CHECK(target == 100);
  CHECK(winners == std::vector<election_key_t>{{1, 3}});
  CHECK(election_rows(merges) == std::vector<election_row_t>{{0, 4, 100}});
}

TEST_CASE("election: originals never survive a coincidence",
          "[intersect][identity]") {
  std::vector<election_key_t> winners;
  tf::buffer<election_row_t> merges;
  const auto target = elect_one({{2, 4}, {2, 8}}, winners, merges);
  CHECK(target == 100);
  CHECK(winners == std::vector<election_key_t>{{2, 4}});
  CHECK(election_rows(merges) ==
        std::vector<election_row_t>{{0, 4, 100}, {0, 8, 100}});
}

TEST_CASE("election: mints follow the caller's coincidence order",
          "[intersect][identity]") {
  std::vector<election_key_t> winners;
  tf::buffer<election_row_t> merges;
  CHECK(elect_one({{1, 6}}, winners, merges) == 100);
  CHECK(elect_one({{0, 2}, {1, 1}}, winners, merges) == 2);
  CHECK(elect_one({{1, 1}, {2, 3}}, winners, merges) == 101);
  CHECK(winners == std::vector<election_key_t>{{1, 6}, {1, 1}});
  CHECK(election_rows(merges) == std::vector<election_row_t>{{0, 3, 101}});
}

TEST_CASE("election: sparse relations elect at ascending roots",
          "[intersect][identity]") {
  // five coincidences, the relations stated so the last-rooted ones come
  // first: {1,4} {1,5} {2,40} | {0,7} {2,41} {1,6} | {0,3} {0,8} and the
  // lone fresh statements {1,2} and {1,9}, one of them stated twice
  auto keys = election_filled<election_key_t>({{1, 9}, {1, 2}, {1, 9}});
  const auto relations =
      election_filled<election_relation_t>({{{{2, 40}, {1, 5}}},
                                            {{{1, 5}, {1, 4}}},
                                            {{{0, 7}, {2, 41}}},
                                            {{{2, 41}, {1, 6}}},
                                            {{{0, 8}, {0, 3}}}});
  std::vector<election_key_t> winners;
  tf::buffer<election_index_t> targets;
  auto merges = election_filled<election_row_t>({{9, 9, 9}});
  tf::intersect::graph::elect_identity_components(
      keys, relations, election_minter{&winners}, targets, merges);

  CHECK(election_rows(keys) == std::vector<election_key_t>{{0, 3},
                                                           {0, 7},
                                                           {0, 8},
                                                           {1, 2},
                                                           {1, 4},
                                                           {1, 5},
                                                           {1, 6},
                                                           {1, 9},
                                                           {2, 40},
                                                           {2, 41}});
  CHECK(winners == std::vector<election_key_t>{{1, 2}, {1, 4}, {1, 9}});
  CHECK(election_rows(targets) ==
        std::vector<election_index_t>{3, 7, 3, 100, 101, 101, 7, 102, 101, 7});
  CHECK(election_rows(merges) ==
        std::vector<election_row_t>{{0, 40, 101}, {0, 41, 7}, {1, 8, 3}});
}

TEST_CASE("election: no candidates elect nothing", "[intersect][identity]") {
  tf::buffer<election_key_t> keys;
  const tf::buffer<election_relation_t> relations;
  std::vector<election_key_t> winners;
  tf::buffer<election_index_t> targets;
  auto merges = election_filled<election_row_t>({{9, 9, 9}});
  tf::intersect::graph::elect_identity_components(
      keys, relations, election_minter{&winners}, targets, merges);
  CHECK(keys.size() == 0);
  CHECK(targets.size() == 0);
  CHECK(merges.size() == 0);
  CHECK(winners.empty());
}
