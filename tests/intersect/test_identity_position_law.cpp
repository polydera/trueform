/**
 * @file test_identity_position_law.cpp
 * @brief THE ONE POSITION LAW over a hand-stated lawful weld set.
 *
 * Created 5 is retired into created 7, and 7's statement reaches 5
 * through 6: the survivor descends from its retiree. Inside 7's
 * evaluation, at every depth, 5 speaks its pre-mutation source; outside
 * it, 5 speaks 7. Every evaluation is finite, whichever end of the weld it
 * starts from. A position here is the sum its statement names, so each
 * answer pins which statements were read.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/intersect/graph/final_identity_point.hpp>
#include <trueform/intersect/graph/plane_identity_collapse.hpp>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using position_law_index_t = int;
using position_law_mint_t =
    tf::intersect::graph::evaluated_identity<position_law_index_t>;

/// Two forms of four vertices each; created 5 retired into created 7.
struct position_law_world {
  std::array<position_law_index_t, 2> vertex_offsets{0, 4};
  tf::buffer<std::array<position_law_index_t, 3>> merges;
  /// Per created identity, the endpoints its statement reads.
  std::vector<std::vector<std::pair<std::int16_t, position_law_index_t>>>
      statements;

  position_law_world() {
    merges.push_back({1, 5, 7});
    statements.resize(9);
    statements[5] = {{0, 3}, {0, 0}};
    statements[6] = {{-1, 5}, {0, 2}};
    statements[7] = {{-1, 6}, {0, 1}};
    statements[8] = {{-1, 5}};
  }

  auto position(std::int16_t tag, position_law_index_t id,
                const position_law_mint_t *mints, int depth) const -> long {
    // a revisit would recurse without bound; stop well before the stack
    REQUIRE(depth < 16);
    return tf::intersect::graph::final_identity_point<long>(
        merges.size() != 0,
        [this](position_law_index_t source, position_law_index_t key) {
          return tf::intersect::graph::plane_merge_target(merges, source, key);
        },
        vertex_offsets, tag, id, mints,
        [&](std::int16_t at_tag, position_law_index_t at,
            const position_law_mint_t *self) -> long {
          if (at_tag >= 0)
            return 1000L * at_tag + at;
          long sum = 10L * at;
          for (const auto &read : statements[std::size_t(at)])
            sum += position(read.first, read.second, self, depth + 1);
          return sum;
        });
  }

  auto position(std::int16_t tag, position_law_index_t id) const -> long {
    return position(tag, id, nullptr, 0);
  }
};

} // namespace

TEST_CASE("position law: a survivor descending from its retiree is finite",
          "[intersect][identity]") {
  const position_law_world world;
  // 5 = 50 + v3 + v0 = 53, read inside 7's evaluation two statements deep
  // 6 = 60 + 53 + v2 = 115
  // 7 = 70 + 115 + v1 = 186
  CHECK(world.position(-1, 7) == 186);
  CHECK(world.position(-1, 6) == 60 + 186 + 2);
  // the retired key speaks its survivor
  CHECK(world.position(-1, 5) == 186);
  // a statement outside the weld reads the survivor
  CHECK(world.position(-1, 8) == 80 + 186);
}

TEST_CASE("position law: an original speaks its own form's vertex",
          "[intersect][identity]") {
  const position_law_world world;
  CHECK(world.position(0, 3) == 3);
  CHECK(world.position(1, 2) == 1002);
}

TEST_CASE("position law: a retired original speaks its survivor",
          "[intersect][identity]") {
  position_law_world world;
  // original (1, 2) is flat vertex 6
  world.merges.clear();
  world.merges.push_back({0, 6, 7});
  world.merges.push_back({1, 5, 7});
  CHECK(world.position(1, 2) == 186);
}
