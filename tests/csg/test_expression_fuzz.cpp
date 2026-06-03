/**
 * @file test_expression_fuzz.cpp
 * @brief Random-tree fuzz over the CSG boolean expression algebra.
 *
 * Generates 5000 random expression trees over 10 operand bits and
 * verifies the compiled evaluator agrees with the tree-walk
 * evaluator on every bitvector. The tree walker is the reference
 * (straightforward switch over the algebra); the compiled
 * (word-mask lowered) form is the path under test.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/csg/expression.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace csg = tf::csg;

namespace {

constexpr int K_OPERANDS = 10;
constexpr int N_TREES = 5000;
constexpr int MAX_DEPTH = 5;

struct bv_t {
  std::uint32_t word;
  auto operator[](std::size_t i) const -> std::uint32_t {
    return i == 0 ? word : 0u;
  }
};

auto random_tree(std::mt19937 &rng, int depth_left) -> csg::expr {
  if (depth_left == 0 || (rng() % 4) == 0)
    return csg::expr{int(rng() % K_OPERANDS)};

  const int kind = rng() % 5;
  switch (kind) {
  case 0: {
    const int n = 1 + int(rng() % 5);
    std::vector<csg::expr> children;
    children.reserve(n);
    for (int i = 0; i < n; ++i)
      children.push_back(random_tree(rng, depth_left - 1));
    return csg::expr{csg::expr::kind::merge, std::move(children)};
  }
  case 1: {
    const int n = 1 + int(rng() % 5);
    std::vector<csg::expr> children;
    children.reserve(n);
    for (int i = 0; i < n; ++i)
      children.push_back(random_tree(rng, depth_left - 1));
    return csg::expr{csg::expr::kind::intersection, std::move(children)};
  }
  case 2: {
    std::vector<csg::expr> children;
    children.reserve(2);
    children.push_back(random_tree(rng, depth_left - 1));
    children.push_back(random_tree(rng, depth_left - 1));
    return csg::expr{csg::expr::kind::difference, std::move(children)};
  }
  case 3: {
    std::vector<csg::expr> children;
    children.reserve(1);
    children.push_back(random_tree(rng, depth_left - 1));
    return csg::expr{csg::expr::kind::complement, std::move(children)};
  }
  default:
    return csg::expr{int(rng() % K_OPERANDS)};
  }
}

} // namespace

TEST_CASE("csg::expr -- random tree walker matches compiled",
          "[csg][expression][fuzz]") {
  std::mt19937 rng(0xC51AC);
  constexpr std::uint32_t N_BV = 1u << K_OPERANDS;

  std::size_t mismatches = 0;
  for (int t = 0; t < N_TREES; ++t) {
    auto tree = random_tree(rng, MAX_DEPTH);
    auto compiled = tree.compile();
    for (std::uint32_t w = 0; w < N_BV; ++w) {
      const bv_t bv{w};
      const bool walker_ans = tree.evaluate(bv);
      const bool compiled_ans = compiled.evaluate(bv);
      if (walker_ans != compiled_ans) {
        INFO("tree=" << t << " bv=" << w
                     << " walker=" << int(walker_ans)
                     << " compiled=" << int(compiled_ans));
        ++mismatches;
        REQUIRE(walker_ans == compiled_ans);
      }
    }
  }
  REQUIRE(mismatches == 0);
}
