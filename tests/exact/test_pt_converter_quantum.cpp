/**
 * @file test_pt_converter_quantum.cpp
 * @brief tf::exact::pt_converter — the quantum a converter carries.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/aabb.hpp>
#include <trueform/core/point.hpp>
#include <trueform/exact/pt_converter.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

TEST_CASE("pt_converter: a converter carries its input's quantum",
          "[exact][converter]") {
  // A float box reaching 1000.5 in magnitude: the house converter's
  // quantum is the ulp of that magnitude on its lattice, rounded up.
  const auto box = tf::make_aabb(tf::point<float, 3>{999.5f, -2.f, 0.f},
                                 tf::point<float, 3>{1000.5f, 2.f, 1.f});
  const auto converter = tf::exact::make_pt_converter<std::int32_t, float>(box);
  const float extent = 1000.5f;
  const float ulp =
      std::nextafter(extent, std::numeric_limits<float>::infinity()) - extent;
  CHECK(converter.quantum() ==
        std::int32_t(std::ceil(double(ulp) * double(converter.scale))));
  CHECK(converter.quantum() > 0);
  // An integral input is exact on its lattice: 0 unless a quantum is stated.
  CHECK(tf::exact::pt_converter<std::int32_t, std::int32_t, 3>{}.quantum() ==
        0);
  CHECK(
      tf::exact::pt_converter<std::int32_t, std::int32_t, 3>{{32}}.quantum() ==
      32);
}
