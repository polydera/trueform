/**
 * @file test_expression.cpp
 * @brief Tests for the CSG boolean expression algebra.
 *
 * Covers walker + compiled evaluators across hand-written algebraic
 * shapes -- merge, intersection, difference, complement, any_of (range
 * and braced-init-list), and mixed leaf-and-subexpression children.
 *
 * Every check evaluates the expression over the 16 possible
 * bitvectors with 4 operand bits and compares against a hand-written
 * reference boolean. Both `expr::evaluator()` (tree walker) and
 * `expr::compile().evaluator()` (word-mask lowered) must agree with
 * the reference.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/csg/expression.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace csg = tf::csg;

namespace {

struct bv4 {
  std::uint32_t word;
  auto operator[](std::size_t i) const -> std::uint32_t {
    (void)i;
    return word;
  }
};

auto bit(std::uint32_t w, int i) -> bool {
  return (w & (std::uint32_t(1) << i)) != 0;
}

template <typename Expr, typename Ref>
void check_against(const Expr &e, Ref ref) {
  auto walk = e.evaluator();
  auto comp = e.compile().evaluator();
  for (std::uint32_t w = 0; w < 16; ++w) {
    const bv4 bv{w};
    INFO("bv=" << w);
    REQUIRE(walk(bv) == ref(w));
    REQUIRE(comp(bv) == ref(w));
  }
}

} // namespace

TEST_CASE("csg::expr algebra agrees with hand-written reference",
          "[csg][expression]") {
  SECTION("merge (union)") {
    check_against(csg::merge(0, 1), [](std::uint32_t w) {
      return bit(w, 0) || bit(w, 1);
    });
  }

  SECTION("intersection") {
    check_against(csg::intersection(0, 1), [](std::uint32_t w) {
      return bit(w, 0) && bit(w, 1);
    });
  }

  SECTION("difference") {
    check_against(csg::difference(0, 1), [](std::uint32_t w) {
      return bit(w, 0) && !bit(w, 1);
    });
  }

  SECTION("complement") {
    check_against(csg::complement(0),
                  [](std::uint32_t w) { return !bit(w, 0); });
  }

  SECTION("any_of (range)") {
    std::array<int, 3> ids{1, 2, 3};
    check_against(csg::any_of(ids), [](std::uint32_t w) {
      return bit(w, 1) || bit(w, 2) || bit(w, 3);
    });
  }

  SECTION("any_of (braced-init-list)") {
    check_against(csg::any_of({1, 2, 3}), [](std::uint32_t w) {
      return bit(w, 1) || bit(w, 2) || bit(w, 3);
    });
  }

  SECTION("all_of (braced-init-list)") {
    check_against(csg::all_of({1, 2, 3}), [](std::uint32_t w) {
      return bit(w, 1) && bit(w, 2) && bit(w, 3);
    });
  }

  SECTION("difference of merge and intersection -- nested") {
    check_against(csg::difference(csg::merge(0, 1),
                                    csg::intersection(2, 3)),
                  [](std::uint32_t w) {
                    return (bit(w, 0) || bit(w, 1)) &&
                           !(bit(w, 2) && bit(w, 3));
                  });
  }

  SECTION("any_of over mixed leaves and sub-expressions") {
    std::vector<csg::expr> mixed;
    mixed.emplace_back(csg::intersection(1, 2));
    mixed.emplace_back(3);
    check_against(csg::any_of(mixed), [](std::uint32_t w) {
      return (bit(w, 1) && bit(w, 2)) || bit(w, 3);
    });
  }
}
