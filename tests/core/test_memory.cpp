/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#include <trueform/core/buffer.hpp>
#include <trueform/core/hash_map.hpp>
#include <trueform/core/hash_set.hpp>
#include <trueform/core/memory.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>

namespace {
template <typename> struct is_tf_allocator : std::false_type {};
template <typename T> struct is_tf_allocator<tf::allocator<T>> : std::true_type {};
} // namespace

TEST_CASE("tf::allocate/tf::deallocate round-trips", "[memory]") {
  int *p = tf::allocate<int>(64);
  REQUIRE(p != nullptr);
  std::iota(p, p + 64, 0);
  REQUIRE(p[63] == 63);
  tf::deallocate<int>(p);
}

TEST_CASE("tf::allocator drives a std::vector", "[memory]") {
  std::vector<int, tf::allocator<int>> v;
  for (int i = 0; i < 1000; ++i) v.push_back(i);
  REQUIRE(v.size() == 1000);
  REQUIRE(v.back() == 999);
  tf::core::std_vector<double> w(10, 1.5);
  REQUIRE(w[9] == 1.5);
  REQUIRE(tf::allocator<int>{} == tf::allocator<int>{});
}

TEST_CASE("tf::hash_map/set use tf::allocator by default", "[memory]") {
  static_assert(is_tf_allocator<tf::hash_map<int, int>::allocator_type>::value,
                "hash_map must default to a tf::allocator");
  static_assert(is_tf_allocator<tf::hash_set<int>::allocator_type>::value,
                "hash_set must default to a tf::allocator");
  tf::hash_map<int, int> m;
  for (int i = 0; i < 500; ++i) m[i] = i * 2;
  REQUIRE(m.at(499) == 998);
  tf::hash_set<int> s{1, 2, 3, 2, 1};
  REQUIRE(s.size() == 3);
}

TEST_CASE("buffer::release yields a tf::deallocate-owned pointer", "[memory]") {
  tf::buffer<int> b;
  b.allocate(256);
  for (int i = 0; i < 256; ++i)
    b[i] = i;
  int *raw = b.release();
  REQUIRE(raw[255] == 255);
  tf::deallocate<int>(raw);
}

TEST_CASE("tf::allocate honors over-alignment", "[memory]") {
  struct alignas(128) slot { double v[4]; };
  static_assert(alignof(slot) == 128);
  slot *p = tf::allocate<slot>(50);
  REQUIRE(reinterpret_cast<std::uintptr_t>(p) % 128 == 0);
  tf::deallocate<slot>(p);

  tf::core::std_vector<slot> v(40);
  REQUIRE(reinterpret_cast<std::uintptr_t>(v.data()) % 128 == 0);
  for (int i = 0; i < 100; ++i) v.push_back(slot{});
  REQUIRE(reinterpret_cast<std::uintptr_t>(v.data()) % 128 == 0);
}
