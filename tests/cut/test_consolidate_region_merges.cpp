/**
 * @file test_consolidate_region_merges.cpp
 * @brief tf::cut::consolidate_region_merges — what its answer means, and
 *        the table it leaves behind.
 *
 * The bool is "the authority CHANGED", not "there are merges": nothing
 * incoming and incoming that is already known both answer no, whatever the
 * table holds. A caller that reads it the other way stops a wave that has
 * work left. The table itself is the contract its single-binary-search
 * consumer needs — sorted by `from`, no chains, no duplicates — and it is
 * reached from a cycle as well as from a chain.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/cut/detail/region_triangulation_types.hpp>
#include <trueform/cut/impl/consolidate_region_merges.hpp>
#include <trueform/intersect/graph/vertex.hpp>

#include <array>
#include <cstddef>

namespace {

using Index = int;
using key_t = std::array<Index, 2>;
using merge_t = tf::cut::detail::region_triangulation_merge<Index>;

constexpr key_t ka{0, 1};
constexpr key_t kb{0, 2};
constexpr key_t kc{0, 3};
constexpr key_t kd{1, 1};

auto make_merge(const key_t &from, const key_t &to_key, Index id) -> merge_t {
  merge_t merge{};
  merge.from = from;
  merge.to_key = to_key;
  merge.to.source = tf::intersect::graph::vertex_source::created;
  merge.to.id = id;
  return merge;
}

auto filled(std::initializer_list<merge_t> merges) -> tf::buffer<merge_t> {
  tf::buffer<merge_t> buffer;
  buffer.reserve(merges.size());
  for (const auto &merge : merges)
    buffer.push_back(merge);
  return buffer;
}

/// The three properties tf::cut::resolve_region_vertex_key depends on.
auto check_consumable(const tf::buffer<merge_t> &merges) -> void {
  for (std::size_t index = 1; index < merges.size(); ++index)
    CHECK(merges[index - 1].from < merges[index].from);
  for (const auto &merge : merges)
    for (const auto &other : merges)
      CHECK(other.from != merge.to_key);
}

} // namespace

TEST_CASE("consolidate_region_merges: nothing incoming is no change, however "
          "much the table already holds",
          "[consolidate_region_merges]") {
  auto merges = filled({make_merge(kb, ka, 7)});
  tf::buffer<merge_t> incoming;

  CHECK_FALSE(tf::cut::consolidate_region_merges<Index>(merges, incoming));
  REQUIRE(merges.size() == 1u);
  CHECK(merges[0].from == kb);
  CHECK(merges[0].to_key == ka);
  CHECK(merges[0].to.id == 7);
}

TEST_CASE("consolidate_region_merges: incoming the table already knows is no "
          "change",
          "[consolidate_region_merges]") {
  tf::buffer<merge_t> merges;
  auto first = filled({make_merge(kb, ka, 7)});
  REQUIRE(tf::cut::consolidate_region_merges<Index>(merges, first));
  REQUIRE(merges.size() == 1u);

  auto again = filled({make_merge(kb, ka, 7)});
  CHECK_FALSE(tf::cut::consolidate_region_merges<Index>(merges, again));
  CHECK(merges.size() == 1u);

  // The same equivalence stated from the other end is the same fact.
  auto reversed = filled({make_merge(ka, kb, 7)});
  CHECK_FALSE(tf::cut::consolidate_region_merges<Index>(merges, reversed));
  REQUIRE(merges.size() == 1u);
  CHECK(merges[0].from == kb);
  CHECK(merges[0].to_key == ka);
  check_consumable(merges);
}

TEST_CASE("consolidate_region_merges: incoming with something new is a change",
          "[consolidate_region_merges]") {
  tf::buffer<merge_t> merges;
  auto first = filled({make_merge(kb, ka, 7)});
  CHECK(tf::cut::consolidate_region_merges<Index>(merges, first));
  CHECK(merges.size() == 1u);

  auto second = filled({make_merge(kd, kc, 9)});
  CHECK(tf::cut::consolidate_region_merges<Index>(merges, second));
  CHECK(merges.size() == 2u);
  check_consumable(merges);

  auto known = filled({make_merge(kd, kc, 9)});
  CHECK_FALSE(tf::cut::consolidate_region_merges<Index>(merges, known));
  CHECK(merges.size() == 2u);
}

TEST_CASE("consolidate_region_merges: a cycle elects one representative",
          "[consolidate_region_merges]") {
  tf::buffer<merge_t> merges;
  auto cycle = filled({make_merge(ka, kb, 2), make_merge(kb, kc, 3),
                       make_merge(kc, ka, 1)});

  CHECK(tf::cut::consolidate_region_merges<Index>(merges, cycle));
  REQUIRE(merges.size() == 2u);
  CHECK(merges[0].from == kb);
  CHECK(merges[1].from == kc);
  CHECK(merges[0].to_key == ka);
  CHECK(merges[1].to_key == ka);
  CHECK(merges[0].to.id == 1);
  CHECK(merges[1].to.id == 1);
  check_consumable(merges);

  // The class is closed: the cycle told again adds nothing.
  auto again = filled({make_merge(ka, kb, 2), make_merge(kb, kc, 3),
                       make_merge(kc, ka, 1)});
  CHECK_FALSE(tf::cut::consolidate_region_merges<Index>(merges, again));
  CHECK(merges.size() == 2u);
}

TEST_CASE("consolidate_region_merges: a chain is flattened onto its "
          "representative",
          "[consolidate_region_merges]") {
  tf::buffer<merge_t> merges;
  auto chain = filled({make_merge(ka, kb, 2), make_merge(kb, kc, 3)});

  CHECK(tf::cut::consolidate_region_merges<Index>(merges, chain));
  REQUIRE(merges.size() == 2u);
  CHECK(merges[0].from == ka);
  CHECK(merges[1].from == kc);
  CHECK(merges[0].to_key == kb);
  CHECK(merges[1].to_key == kb);
  CHECK(merges[0].to.id == 2);
  CHECK(merges[1].to.id == 2);
  check_consumable(merges);
}
