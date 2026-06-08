/**
 * @file test_int128_msvc.cpp
 * @brief MSVC-only cross-check of tf::exact::int128 / uint128 against the
 *        bundled std::_Signed128 / std::_Unsigned128.
 *
 * On MSVC there is no native __int128, so tf::exact provides a hand-rolled
 * 128-bit integer. This test sweeps a large set of operand pairs (hand-picked
 * corners, a small-magnitude grid, and a deterministic random stream) through
 * every operator and checks bit-for-bit agreement with the STL's bundled
 * 128-bit integers, which are the reference. On every other compiler the types
 * are native __int128 and this test is a no-op.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>

#if defined(_MSC_VER)

#include <trueform/exact/int128.hpp>

#include <__msvc_int128.hpp>

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using tf_u = tf::exact::uint128;
using tf_i = tf::exact::int128;
using ms_u = std::_Unsigned128;
using ms_i = std::_Signed128;

// A 128-bit value as its two 64-bit limbs. The common form for comparing the
// two implementations independently of their internal storage.
struct limbs {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
  auto operator==(const limbs &o) const -> bool {
    return lo == o.lo && hi == o.hi;
  }
};

// Decompose any of the four 128-bit types into its limbs. Works for signed
// types too: an arithmetic right shift by 64 leaves the high limb in the low
// word, and the cast truncates to it.
template <typename T> auto split(const T &v) -> limbs {
  return {static_cast<std::uint64_t>(v), static_cast<std::uint64_t>(v >> 64)};
}

// Build any of the four types from a raw bit pattern.
template <typename T> auto make(const limbs &l) -> T {
  return T(l.lo) | (T(l.hi) << 64);
}

// The first operator on which the two implementations disagree, if any.
struct failure {
  bool any = false;
  const char *op = "";
  limbs a, b, tf, ms;
};

auto to_hex(const limbs &l) -> std::string {
  char buf[40];
  std::snprintf(buf, sizeof buf, "0x%016llx%016llx",
                static_cast<unsigned long long>(l.hi),
                static_cast<unsigned long long>(l.lo));
  return buf;
}

auto describe(const failure &f) -> std::string {
  return std::string("operator '") + f.op + "': a=" + to_hex(f.a) +
         " b=" + to_hex(f.b) + " trueform=" + to_hex(f.tf) +
         " msvc=" + to_hex(f.ms);
}

// Run every operator on one ordered pair and record the first disagreement.
// TF is the trueform type, MS the matching MSVC reference type (both unsigned
// or both signed, so operator semantics line up).
template <typename TF, typename MS>
auto sweep_pair(failure &f, const limbs &a, const limbs &b) -> void {
  if (f.any)
    return;
  const TF ta = make<TF>(a), tb = make<TF>(b);
  const MS ma = make<MS>(a), mb = make<MS>(b);

  auto val = [&](const char *op, limbs t, limbs m) {
    if (!f.any && !(t == m))
      f = {true, op, a, b, t, m};
  };
  auto cmp = [&](const char *op, bool t, bool m) {
    if (!f.any && t != m)
      f = {true, op, a, b, limbs{t}, limbs{m}};
  };

  val("+", split(ta + tb), split(ma + mb));
  val("-", split(ta - tb), split(ma - mb));
  val("*", split(ta * tb), split(ma * mb));
  val("&", split(ta & tb), split(ma & mb));
  val("|", split(ta | tb), split(ma | mb));
  val("^", split(ta ^ tb), split(ma ^ mb));
  if (b.lo != 0 || b.hi != 0) {
    val("/", split(ta / tb), split(ma / mb));
    val("%", split(ta % tb), split(ma % mb));
  }
  cmp("==", ta == tb, ma == mb);
  cmp("!=", ta != tb, ma != mb);
  cmp("<", ta < tb, ma < mb);
  cmp(">", ta > tb, ma > mb);
  cmp("<=", ta <= tb, ma <= mb);
  cmp(">=", ta >= tb, ma >= mb);
  val("~", split(~ta), split(~ma));
  val("unary-", split(-ta), split(-ma));

  static const unsigned shifts[] = {0, 1, 7, 31, 63, 64, 65, 95, 127};
  for (unsigned s : shifts) {
    val("<<", split(ta << s), split(ma << s));
    val(">>", split(ta >> s), split(ma >> s));
  }

  val("unary+", split(+ta), split(+ma));

  // increment and decrement, pre and post, on copies (wraparound at the
  // corners is the interesting case)
  {
    TF t = ta;
    MS m = ma;
    val("pre++", split(++t), split(++m));
    val("pre++ state", split(t), split(m));
  }
  {
    TF t = ta;
    MS m = ma;
    val("pre--", split(--t), split(--m));
    val("pre-- state", split(t), split(m));
  }
  {
    TF t = ta;
    MS m = ma;
    val("post++ result", split(t++), split(m++));
    val("post++ state", split(t), split(m));
  }
  {
    TF t = ta;
    MS m = ma;
    val("post-- result", split(t--), split(m--));
    val("post-- state", split(t), split(m));
  }

  // compound assignment, on copies
  auto compound = [&](const char *op, auto f_tf, auto f_ms) {
    TF t = ta;
    MS m = ma;
    f_tf(t);
    f_ms(m);
    val(op, split(t), split(m));
  };
  compound("+=", [&](TF &t) { t += tb; }, [&](MS &m) { m += mb; });
  compound("-=", [&](TF &t) { t -= tb; }, [&](MS &m) { m -= mb; });
  compound("*=", [&](TF &t) { t *= tb; }, [&](MS &m) { m *= mb; });
  compound("&=", [&](TF &t) { t &= tb; }, [&](MS &m) { m &= mb; });
  compound("|=", [&](TF &t) { t |= tb; }, [&](MS &m) { m |= mb; });
  compound("^=", [&](TF &t) { t ^= tb; }, [&](MS &m) { m ^= mb; });
  compound("<<=", [&](TF &t) { t <<= 64; }, [&](MS &m) { m <<= 64; });
  compound(">>=", [&](TF &t) { t >>= 64; }, [&](MS &m) { m >>= 64; });
  if (b.lo != 0 || b.hi != 0) {
    compound("/=", [&](TF &t) { t /= tb; }, [&](MS &m) { m /= mb; });
    compound("%=", [&](TF &t) { t %= tb; }, [&](MS &m) { m %= mb; });
  }
}

// Hand-picked corner bit patterns: zero, one, all-ones, limb boundaries, sign
// bits, small magnitudes (the multiply fast paths), full scattered words, and
// powers of two with their neighbours straddling the 32/64/96/128-bit
// boundaries (carries, borrows, and multiply crossings live here).
auto corners() -> const std::vector<limbs> & {
  static const std::vector<limbs> v = [] {
    std::vector<limbs> out = {
        {0, 0},        {1, 0},        {2, 0},        {3, 0},
        {7, 0},        {10, 0},       {1000, 0},     {~0ull, 0},
        {0, 1},        {0, ~0ull},    {~0ull, ~0ull},
        {0x8000000000000000ull, 0},  {0, 0x8000000000000000ull},
        {0x7fffffffffffffffull, 0},  {~0ull, 0x7fffffffffffffffull},
        {1, ~0ull},    {~0ull, 1},    {0xffffffffull, 0},
        {0xffffffff00000000ull, 0},  {0xffffffffull, 0xffffffff00000000ull},
        {0x123456789abcdef0ull, 0xfedcba9876543210ull},
        {0xdeadbeefcafebabeull, 0x0123456789abcdefull},
        {0xaaaaaaaaaaaaaaaaull, 0x5555555555555555ull},
    };
    for (unsigned k : {1u, 31u, 32u, 33u, 47u, 48u, 62u, 63u, 64u, 65u, 66u,
                       95u, 96u, 97u, 126u, 127u}) {
      limbs p = (k < 64) ? limbs{1ull << k, 0} : limbs{0, 1ull << (k - 64)};
      limbs pm1 = p; // p - 1, borrowing across the boundary if needed
      if (p.lo == 0) {
        pm1.lo = ~0ull;
        pm1.hi = p.hi - 1;
      } else {
        pm1.lo = p.lo - 1;
      }
      limbs pp1 = p; // p + 1, carrying across the boundary if needed
      if (p.lo == ~0ull) {
        pp1.lo = 0;
        pp1.hi = p.hi + 1;
      } else {
        pp1.lo = p.lo + 1;
      }
      out.push_back(p);
      out.push_back(pm1);
      out.push_back(pp1);
    }
    return out;
  }();
  return v;
}

template <typename TF, typename MS> auto run_sweep() -> failure {
  failure f;

  // corner against corner
  for (const auto &a : corners())
    for (const auto &b : corners())
      sweep_pair<TF, MS>(f, a, b);

  // small-magnitude grid, including small negatives (high limb all ones),
  // to hammer the i64-fast-path multiply and the division bit loop
  for (std::uint64_t i = 0; i < 64 && !f.any; ++i)
    for (std::uint64_t j = 0; j < 64; ++j) {
      sweep_pair<TF, MS>(f, {i, 0}, {j, 0});
      sweep_pair<TF, MS>(f, {~i, ~0ull}, {j, 0});
      sweep_pair<TF, MS>(f, {~i, ~0ull}, {~j, ~0ull});
    }

  // deterministic random stream over the full 128-bit space, with some
  // single-limb operands mixed in
  std::mt19937_64 rng(0x9e3779b97f4a7c15ull);
  for (int k = 0; k < 100000 && !f.any; ++k) {
    limbs a{rng(), rng()};
    limbs b{rng(), rng()};
    if ((k & 3) == 0)
      b.hi = 0; // exercise the narrow-operand paths
    if ((k & 7) == 0)
      a.hi = 0;
    sweep_pair<TF, MS>(f, a, b);
  }
  return f;
}

} // namespace

TEST_CASE("uint128 matches std::_Unsigned128 over a large sweep",
          "[int128][msvc]") {
  failure f = run_sweep<tf_u, ms_u>();
  if (f.any)
    UNSCOPED_INFO(describe(f));
  REQUIRE_FALSE(f.any);
}

TEST_CASE("int128 matches std::_Signed128 over a large sweep",
          "[int128][msvc]") {
  failure f = run_sweep<tf_i, ms_i>();
  if (f.any)
    UNSCOPED_INFO(describe(f));
  REQUIRE_FALSE(f.any);
}

#else

TEST_CASE("int128 cross-check against the bundled type is MSVC only",
          "[int128]") {
  SUCCEED("tf::exact::int128 is native __int128 here; nothing to cross-check");
}

#endif
